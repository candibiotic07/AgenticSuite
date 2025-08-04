"""
AI-Powered Meeting Notes Automation

This agent stores data in organized directories:
- meetingDATAtemp/
  ├── data/          # Meeting data and processing logs
  └── logs/          # Log files
- token.json         # Centralized authentication (root directory)

The agent will automatically migrate existing files and handle OAuth scope requirements.
"""

import os
import re
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai  # For Gemini API

# Load environment variables from a .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; fall back to system environment variables
    pass

# Create main data directory structure
BASE_DATA_DIR = "meetingDATAtemp"
os.makedirs(os.path.join(BASE_DATA_DIR, "logs"), exist_ok=True)    # Log files
os.makedirs(os.path.join(BASE_DATA_DIR, "data"), exist_ok=True)    # Meeting data
# Note: Authentication uses centralized token.json in root directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DATA_DIR, "logs", "meeting_notes.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# If you prefer OpenAI instead of Gemini, uncomment below:
# import openai

class MeetingNotesAutomation:
    """
    Automates the process of generating AI-powered meeting notes from Google Meet recordings.
    
    The automation can be configured to fetch meetings from a customizable time range
    (default: 30 days back from current date).
    """
    
    # Required OAuth2 scopes for full functionality
    SCOPES = [
        'https://www.googleapis.com/auth/calendar.readonly',         # Read calendar events
        'https://www.googleapis.com/auth/meetings.space.readonly',   # Read Meet spaces & conference records
        'https://www.googleapis.com/auth/meetings.space.created',    # Access created Meet spaces
        'https://www.googleapis.com/auth/documents',                 # Create/edit Google Docs
        'https://www.googleapis.com/auth/drive.file',               # Access specific Drive files
        'https://www.googleapis.com/auth/gmail.send',               # Send emails
    ]
    
    def __init__(self, credentials_file=None, gemini_api_key=None, days_back=30):
        """
        Initialize the automation with necessary API clients.
        
        Args:
            credentials_file (str, optional): Path to OAuth credentials file. Defaults to "token.json".
            gemini_api_key (str, optional): Gemini API key. Defaults to GEMINI_API_KEY environment variable.
            days_back (int, optional): Number of days back to fetch meetings from. Defaults to 30.
        """
        # Use centralized token in root directory
        if credentials_file is None:
            credentials_file = "token.json"
        
        # Resolve Gemini API key: environment variable as fallback
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        # Set the configurable time range for fetching meetings
        self.days_back = days_back

        # Migrate existing files and authenticate
        self._migrate_existing_files()
        self.creds = self._authenticate(credentials_file)
        
        # Initialize Google API services
        self.calendar_service = build('calendar', 'v3', credentials=self.creds)
        # The public Meet REST discovery document is currently available.
        # Using 'v2' and providing the discovery URL is the correct approach.
        discovery_url = "https://meet.googleapis.com/$discovery/rest?version=v2"
        self.meet_service = build('meet', 'v2', credentials=self.creds, discoveryServiceUrl=discovery_url, static_discovery=False)
        self.docs_service = build('docs', 'v1', credentials=self.creds)
        self.drive_service = build('drive', 'v3', credentials=self.creds)
        self.gmail_service = build('gmail', 'v1', credentials=self.creds)
        
        # Initialize Gemini AI (or your preferred LLM)
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.gemini_model = None
            logger.warning("GEMINI_API_KEY not found. AI summary generation will be disabled.")
            print("⚠️  GEMINI_API_KEY not found. AI summary generation will be disabled.")
        
        logger.info("Meeting Notes Automation initialized successfully")
        logger.info(f"Data directory: {BASE_DATA_DIR}")
        logger.info(f"  - Logs: {os.path.join(BASE_DATA_DIR, 'logs')}")
        logger.info(f"  - Data: {os.path.join(BASE_DATA_DIR, 'data')}")  
        logger.info("Authentication: Centralized token.json in root directory")
        logger.info(f"Meeting fetch range: {self.days_back} days back from current date")
    
    def _migrate_existing_files(self):
        """Migrate existing files to new directory structure"""
        # Note: token.json stays in root directory for centralized access by all agents
        pass  # No files to migrate for meeting agent currently
    
    def _authenticate(self, credentials_file):
        """Load and validate credentials with proper scopes."""
        creds = None
        
        # Load existing credentials
        if os.path.exists(credentials_file):
            creds = Credentials.from_authorized_user_file(credentials_file, self.SCOPES)
        
        # Check if credentials are valid and have proper scopes
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed credentials
                with open(credentials_file, 'w') as token:
                    token.write(creds.to_json())
                logger.info("Refreshed expired credentials")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                creds = None
        
        # Check if we have valid credentials with required scopes
        if not creds or not creds.valid:
            raise Exception(
                f"""
❌ Invalid or missing credentials!

Please ensure you have:
1. Valid token.json file in: {credentials_file}
2. Token has the required OAuth scopes for Google Meet API

Required scopes:
{chr(10).join(f'  - {scope}' for scope in self.SCOPES)}

To fix this:
1. Delete your existing token.json file
2. Re-run the OAuth setup with updated scopes
3. Make sure to enable Google Meet API in Google Cloud Console

The error you're seeing (ACCESS_TOKEN_SCOPE_INSUFFICIENT) means your current
token doesn't have permission to access Google Meet conference records.
                """.strip()
            )
        
        # Validate scopes (if available in credentials)
        if hasattr(creds, 'scopes') and creds.scopes:
            missing_scopes = []
            for required_scope in self.SCOPES:
                if required_scope not in creds.scopes:
                    missing_scopes.append(required_scope)
            
            if missing_scopes:
                logger.warning(f"Missing scopes: {missing_scopes}")
                print(f"⚠️  Warning: Some required scopes may be missing: {missing_scopes}")
        
        logger.info("Authentication successful")
        return creds
    
    def list_recent_meetings(self, max_results=10):
        """List recent meetings from the user's calendar."""
        print(f"\n🗓️  Fetching your recent meetings (last {self.days_back} days)...")
        
        # Calculate time range using configurable days_back
        now = datetime.now(timezone.utc)
        time_min = (now - timedelta(days=self.days_back)).isoformat()
        time_max = now.isoformat()
        
        try:
            events_result = self.calendar_service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            # Include any meeting that has already started (even if it hasn't officially ended yet).
            # This captures ad-hoc Meet calls or meetings that ran past their scheduled time.
            relevant = []
            for event in events:
                start_str = event.get('start', {}).get('dateTime', event.get('start', {}).get('date'))
                if not start_str:
                    continue  # Skip malformed events

                try:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback for all-day events that only have a date.
                    start_time = datetime.fromisoformat(start_str + 'T00:00:00+00:00')

                if start_time <= now:
                    relevant.append(event)

            # Show most recent first
            return relevant[::-1]
            
        except HttpError as e:
            print(f"Error fetching calendar events: {e}")
            return []
    
    def display_meetings_for_selection(self, meetings):
        """Display meetings and let user select one."""
        print("\n📋 Please select the meeting to process:")
        print("-" * 50)
        
        for i, meeting in enumerate(meetings, 1):
            start_str = meeting.get('start', {}).get('dateTime', meeting.get('start', {}).get('date'))
            end_str = meeting.get('end', {}).get('dateTime', meeting.get('end', {}).get('date'))
            
            if start_str:
                start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                
                # Calculate how long ago it ended
                time_ago = datetime.now(timezone.utc) - end_time
                if time_ago.days > 0:
                    ago_str = f"{time_ago.days} day{'s' if time_ago.days > 1 else ''} ago"
                else:
                    hours = time_ago.seconds // 3600
                    if hours > 0:
                        ago_str = f"{hours} hour{'s' if hours > 1 else ''} ago"
                    else:
                        minutes = time_ago.seconds // 60
                        ago_str = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                
                title = meeting.get('summary', 'Untitled Meeting')
                print(f"[{i}] {title} (Ended: {ago_str})")
                
                # Show attendees count
                attendees = meeting.get('attendees', [])
                if attendees:
                    print(f"    Attendees: {len(attendees)} people")
        
        print("-" * 50)
        
        while True:
            try:
                choice = input("Enter number: ")
                index = int(choice) - 1
                if 0 <= index < len(meetings):
                    return meetings[index]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def process_meeting_by_index(self, meeting_index, progress_callback=None):
        """Process a specific meeting by index (web-compatible version)."""
        def update_progress(step, current, total, message):
            if progress_callback:
                progress_callback(step, current, total, message)
            print(f"[{current}/{total}] {message}")
        
        update_progress("Initializing", 0, 6, "🤖 Welcome to the AI Meeting Notes Assistant!")
        
        # Step 1: List recent meetings  
        update_progress("Loading meetings", 1, 6, "📅 Loading recent meetings...")
        meetings = self.list_recent_meetings()
        if not meetings:
            update_progress("Error", 1, 6, "❌ No recent meetings found.")
            return None
        
        # Step 2: Select meeting by index
        if meeting_index < 0 or meeting_index >= len(meetings):
            update_progress("Error", 1, 6, f"❌ Invalid meeting index: {meeting_index}")
            return None
            
        selected_meeting = meetings[meeting_index]
        meeting_title = selected_meeting.get('summary', 'Untitled Meeting')
        update_progress("Meeting selected", 1, 6, f"✅ Selected: {meeting_title}")
        
        # Step 3: Find conference record(s)
        update_progress("Finding recording", 2, 6, "🔍 Looking for conference record...")
        conference_record = self.find_conference_record(selected_meeting)

        # If initial record found but no transcript later, we'll look at siblings.
        records_for_code = []
        meeting_code = None
        if 'conferenceData' in selected_meeting and selected_meeting['conferenceData'].get('conferenceId'):
            meeting_code = selected_meeting['conferenceData']['conferenceId']
        elif 'hangoutLink' in selected_meeting:
            meeting_code = selected_meeting['hangoutLink'].split('/')[-1]

        if meeting_code:
            records_for_code = self._list_conference_records(meeting_code)
        if conference_record and conference_record not in records_for_code:
            records_for_code.insert(0, conference_record)

        # Try to pick record with a ready transcript first
        chosen_record, transcript = self._pick_record_with_transcript(records_for_code)

        # If none ready, fall back to waiting on most recent record
        if not transcript:
            record_to_wait = conference_record or (records_for_code[-1] if records_for_code else None)
            if not record_to_wait:
                update_progress("Error", 2, 6, "❌ Could not find conference record for this meeting.")
                return None
            update_progress("Waiting for transcript", 3, 6, "⏳ Waiting for transcript to be ready...")
            transcript = self.wait_for_transcript(record_to_wait['name'])
            if not transcript:
                update_progress("Error", 3, 6, "❌ Transcript not available.")
                return None
            chosen_record = record_to_wait
        
        # Step 4: Fetch transcript entries
        update_progress("Fetching transcript", 3, 6, "📄 Fetching transcript content...")
        transcript_text = self.fetch_transcript_entries(transcript['name'])
        if not transcript_text:
            update_progress("Error", 3, 6, "❌ Failed to fetch transcript content.")
            return None
        
        # Step 5: Generate AI summary
        update_progress("Analyzing content", 4, 6, "🤖 Generating AI summary...")
        ai_summary = self.generate_ai_summary(transcript_text, meeting_title)
        
        # Step 6: Create Google Doc
        update_progress("Creating document", 5, 6, "📝 Creating Google Doc...")
        document_id = self.create_google_doc(meeting_title, ai_summary, selected_meeting)
        
        # Step 7: Share document with attendees
        update_progress("Sharing document", 6, 6, "📧 Sharing document with attendees...")
        attendee_emails = [att.get('email') for att in selected_meeting.get('attendees', []) if att.get('email')]

        # Fallback: include organizer if attendees list is empty
        if not attendee_emails:
            organizer_email = selected_meeting.get('organizer', {}).get('email')
            if organizer_email:
                attendee_emails.append(organizer_email)
        doc_link = self.share_document(document_id, attendee_emails)
        
        # Step 8: Send email notifications
        self.send_email_notifications(attendee_emails, meeting_title, doc_link)
        
        update_progress("Completed", 6, 6, f"✨ Success! Meeting notes created and shared.")
        
        return {
            'success': True,
            'meeting_title': meeting_title,
            'document_link': doc_link,
            'attendee_emails': attendee_emails
        }
    
    def find_conference_record(self, calendar_event):
        """Find the Google Meet conference record for a calendar event."""
        print(f"\n🔍 Looking for conference record...")
        
        # Attempt to get the Meet meeting code first. It can live in different places
        meeting_code = None
        if 'conferenceData' in calendar_event and calendar_event['conferenceData'].get('conferenceId'):
            meeting_code = calendar_event['conferenceData']['conferenceId']
        elif 'hangoutLink' in calendar_event and calendar_event['hangoutLink']:
            # hangoutLink looks like https://meet.google.com/abc-defg-hij
            meeting_code = calendar_event['hangoutLink'].split('/')[-1]

        if not meeting_code:
            print("No Google Meet link found in this calendar event.")
            return None

        try:
            # Use meeting_code filter which is supported by the Meet REST API
            filter_str = f"meeting_code=\"{meeting_code}\""

            response = self.meet_service.conferenceRecords().list(
                filter=filter_str
            ).execute()

            records = response.get('conferenceRecords', [])
            if records:
                # Pick the most recent record (last item)
                record = sorted(records, key=lambda r: r.get('startTime', ''), reverse=True)[0]
                print(f"✅ Found conference record: {record['name']}")
                return record
            else:
                print("No conference record found yet. It may still be processing.")
                return None

        except HttpError as e:
            if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
                print("❌ Authentication Error: Your token doesn't have Google Meet API permissions!")
                print("\n🔧 To fix this:")
                print("1. Delete your existing token.json file")
                print("2. Re-run OAuth setup with updated scopes")
                print("3. Make sure Google Meet API is enabled in Google Cloud Console")
                print(f"\nRequired scopes include: {', '.join(self.SCOPES)}")
                logger.error("ACCESS_TOKEN_SCOPE_INSUFFICIENT - token needs Meet API scopes")
            else:
                print(f"Error searching for conference record: {e}")
                logger.error(f"Error searching for conference record: {e}")
            return None

    # ------------------------------------------------------------------
    # NEW HELPERS -------------------------------------------------------
    # ------------------------------------------------------------------

    def _list_conference_records(self, meeting_code: str):
        """Return all conferenceRecords for a given meeting code (may be multiple)."""
        try:
            resp = self.meet_service.conferenceRecords().list(
                filter=f'meeting_code="{meeting_code}"'
            ).execute()
            return resp.get('conferenceRecords', [])
        except HttpError as e:
            if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in str(e):
                logger.error("ACCESS_TOKEN_SCOPE_INSUFFICIENT in _list_conference_records")
            else:
                logger.error(f"Error listing conference records: {e}")
            return []

    def _pick_record_with_transcript(self, records: list):
        """Return a record that already has at least one transcript (or None)."""
        for rec in records:
            try:
                tr_resp = self.meet_service.conferenceRecords().transcripts().list(
                    parent=rec['name']
                ).execute()
                if tr_resp.get('transcripts'):
                    return rec, tr_resp['transcripts'][0]
            except HttpError:
                continue
        return None, None
    
    def wait_for_transcript(self, conference_record_name, max_wait_minutes=30):
        """Poll for transcript availability."""
        print(f"\n⏳ Waiting for transcript to become available...")
        print(f"This may take a few minutes after the meeting ends...")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        check_interval = 60  # Check every minute

        # Try immediate fetch first (sometimes transcript is already available).
        try:
            initial_resp = self.meet_service.conferenceRecords().transcripts().list(parent=conference_record_name).execute()
            if initial_resp.get('transcripts'):
                print("✅ Transcript found immediately!")
                return initial_resp['transcripts'][0]
        except HttpError as e:
            print(f"Error checking for transcript: {e}")

        while (time.time() - start_time) < max_wait_seconds:
            try:
                # List transcripts for this conference
                response = self.meet_service.conferenceRecords().transcripts().list(
                    parent=conference_record_name
                ).execute()
                
                transcripts = response.get('transcripts', [])
                if transcripts:
                    print(f"✅ Transcript found!")
                    return transcripts[0]  # Return the first (usually only) transcript
                
                # Calculate time waited
                waited_minutes = int((time.time() - start_time) / 60)
                print(f"⏱️  Transcript not yet available. Waited {waited_minutes} minutes. Checking again in 1 minute...")
                time.sleep(check_interval)
                
            except HttpError as e:
                print(f"Error checking for transcript: {e}")
                time.sleep(check_interval)
        
        print(f"❌ Transcript not available after {max_wait_minutes} minutes.")
        return None
    
    def fetch_transcript_entries(self, transcript_name):
        """Fetch all transcript entries and format them."""
        print(f"\n📝 Fetching transcript entries...")
        
        all_entries = []
        page_token = None
        
        try:
            while True:
                # Fetch transcript entries with pagination
                request = self.meet_service.conferenceRecords().transcripts().entries().list(
                    parent=transcript_name,
                    pageSize=100,
                    pageToken=page_token
                )
                response = request.execute()
                
                entries = response.get('transcriptEntries', [])
                all_entries.extend(entries)
                
                page_token = response.get('nextPageToken')
                if not page_token:
                    break
            
            print(f"✅ Retrieved {len(all_entries)} transcript entries")
            return self._format_transcript(all_entries)
            
        except HttpError as e:
            print(f"Error fetching transcript entries: {e}")
            return ""
    
    def _format_transcript(self, entries):
        """Format transcript entries into readable text."""
        formatted_lines = []
        
        for entry in entries:
            # Extract timestamp
            raw_start = entry.get('startTime', '0s')
            # The Meet API can return either a duration like "123.45s" or an ISO timestamp.
            if raw_start.endswith('s'):
                # Duration in seconds.
                sec_val = float(raw_start.rstrip('s'))
                minutes = int(sec_val) // 60
                seconds = int(sec_val) % 60
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
            else:
                # ISO 8601 timestamp (e.g., 2025-07-24T08:31:06.391Z)
                try:
                    dt = datetime.fromisoformat(raw_start.replace('Z', '+00:00'))
                    timestamp = f"[{dt.strftime('%H:%M:%S')}]"
                except ValueError:
                    timestamp = "[--:--]"
            
            # Extract speaker and text
            participant_field = entry.get('participant')
            if isinstance(participant_field, dict):
                speaker_name = participant_field.get('name') or participant_field.get('displayName', 'Unknown Speaker')
            elif isinstance(participant_field, str):
                # API sometimes returns the participant resource name string; use last part as fallback label.
                speaker_name = participant_field.split('/')[-1]
            else:
                speaker_name = 'Unknown Speaker'
            text = entry.get('text', '')
            
            # Format the line
            formatted_lines.append(f"{timestamp} {speaker_name}: {text}")
        
        return "\n".join(formatted_lines)
    
    def generate_ai_summary(self, transcript_text, meeting_title):
        """Generate AI summary using Gemini (or your preferred LLM)."""
        if not self.gemini_model:
            print("⚠️  No AI model configured. Skipping AI summary generation.")
            return None
        
        print(f"\n🤖 Generating AI summary with Gemini...")
        
        prompt = f"""
        You are an AI assistant that creates structured, professional meeting notes.
        Based on the meeting transcript below, generate the notes.

        Meeting Title: {meeting_title}

        **Output Format Rules (Strictly Follow):**
        1.  Start each section with a specific markdown header on its own line:
            - `## Executive Summary`
            - `## Key Decisions`
            - `## Action Items`
            - `## Important Points`
        2.  For "Executive Summary", provide a concise paragraph.
        3.  For "Key Decisions" and "Important Points", create a bulleted list. Each item MUST start with `* ` (an asterisk followed by a space).
        4.  For "Action Items", create a checklist. Each item MUST start with `- [ ] ` (a hyphen, space, open bracket, space, close bracket, space). Include assigned owners if mentioned.
        5.  Do not add any other text, titles, or formatting outside of this structure.

        **TRANSCRIPT:**
        {transcript_text}
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating AI summary: {e}")
            return None

    def _parse_ai_summary(self, summary_text: str) -> Dict[str, Any]:
        """Parses the AI-generated markdown summary into a structured dictionary."""
        sections = {
            "Executive Summary": "",
            "Key Decisions": [],
            "Action Items": [],
            "Important Points": []
        }
        
        if not summary_text:
            return sections

        # Split content by the headers
        parts = re.split(r'##\s*(Executive Summary|Key Decisions|Action Items|Important Points)', summary_text)
        
        if len(parts) < 2:
            # Fallback if splitting fails, return as-is under summary
            sections['Executive Summary'] = summary_text.strip()
            return sections

        # The regex split gives us ['', 'Header1', 'Content1', 'Header2', 'Content2', ...]
        i = 1
        while i < len(parts):
            header = parts[i].strip() if parts[i] else ''
            content = parts[i+1].strip() if i + 1 < len(parts) else ''
            
            if header == "Executive Summary":
                sections["Executive Summary"] = content
            elif header in ["Key Decisions", "Important Points"]:
                # Split by newline and filter out empty lines, then strip the bullet point marker
                items = [line.strip()[2:] for line in content.split('\n') if line.strip().startswith('* ')]
                sections[header] = items
            elif header == "Action Items":
                items = [line.strip()[6:] for line in content.split('\n') if line.strip().startswith('- [ ] ')]
                sections[header] = items
            
            i += 2
            
        return sections

    def create_google_doc(self, meeting_title, content, calendar_event):
        """Create a formatted Google Doc with the meeting notes."""
        print(f"\n📄 Creating Google Doc...")
        
        doc_title_filename = f"Meeting Notes: {meeting_title}"
        document = self.docs_service.documents().create(body={'title': doc_title_filename}).execute()
        document_id = document['documentId']
        print(f"✅ Created document: {doc_title_filename}")

        # --- Start building requests ---
        requests = []
        cursor = 1

        # 1. In-document Title (just the meeting name)
        requests.append({'insertText': {'location': {'index': cursor}, 'text': f"{meeting_title}\n"}})
        requests.append({'updateParagraphStyle': {
            'range': {'startIndex': cursor, 'endIndex': cursor + len(meeting_title)},
            'paragraphStyle': {'namedStyleType': 'TITLE'},
            'fields': 'namedStyleType'
        }})
        cursor += len(meeting_title) + 1

        # 2. Metadata (Date and Attendees) - not as a heading
        start_str = calendar_event.get('start', {}).get('dateTime', '')
        if start_str:
            meeting_date = datetime.fromisoformat(start_str.replace('Z', '+00:00')).strftime('%B %d, %Y at %I:%M %p')
            date_text = f"Date: {meeting_date}\n"
            requests.append({'insertText': {'location': {'index': cursor}, 'text': date_text}})
            cursor += len(date_text)

        attendees = calendar_event.get('attendees', [])
        if attendees:
            attendee_names = [att.get('email', '') for att in attendees]
            attendees_text = f"Attendees: {', '.join(attendee_names)}\n\n"
            requests.append({'insertText': {'location': {'index': cursor}, 'text': attendees_text}})
            cursor += len(attendees_text)
        else:
            requests.append({'insertText': {'location': {'index': cursor}, 'text': '\n'}})
            cursor += 1
            
        # 3. Parse AI content and insert formatted sections
        if content:
            parsed_content = self._parse_ai_summary(content)

            section_order = ["Executive Summary", "Key Decisions", "Action Items", "Important Points"]
            for header in section_order:
                section_content = parsed_content.get(header)
                if not section_content:
                    continue

                # Insert header and style it
                header_text = f"{header}\n"
                requests.append({'insertText': {'location': {'index': cursor}, 'text': header_text}})
                requests.append({'updateParagraphStyle': {
                    'range': {'startIndex': cursor, 'endIndex': cursor + len(header) },
                    'paragraphStyle': {'namedStyleType': 'HEADING_2'},
                    'fields': 'namedStyleType'
                }})
                cursor += len(header_text)

                # Insert content based on type
                if header == "Executive Summary":
                    body_text = f"{section_content}\n\n"
                    requests.append({'insertText': {'location': {'index': cursor}, 'text': body_text}})
                    cursor += len(body_text)
                
                elif header in ["Key Decisions", "Important Points"]:
                    for item in section_content:
                        item_text = f"{item}\n"
                        requests.append({'insertText': {'location': {'index': cursor}, 'text': item_text}})
                        # Apply bullet point
                        requests.append({'createParagraphBullets': {
                            'range': {'startIndex': cursor, 'endIndex': cursor + len(item) },
                            'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
                        }})
                        cursor += len(item_text)
                    requests.append({'insertText': {'location': {'index': cursor}, 'text': '\n'}}) # Space after list
                    cursor += 1

                elif header == "Action Items":
                    for item in section_content:
                        item_text = f"{item}\n"
                        requests.append({'insertText': {'location': {'index': cursor}, 'text': item_text}})
                        # Apply checkbox
                        requests.append({'createParagraphBullets': {
                            'range': {'startIndex': cursor, 'endIndex': cursor + len(item) },
                            'bulletPreset': 'BULLET_CHECKBOX'
                        }})
                        cursor += len(item_text)
                    requests.append({'insertText': {'location': {'index': cursor}, 'text': '\n'}}) # Space after list
                    cursor += 1
        else:
            requests.append({'insertText': {'location': {'index': cursor}, 'text': "No AI summary available."}})

        if requests:
            self.docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
        
        return document_id
    
    def share_document(self, document_id, attendee_emails):
        """Share the document with meeting attendees."""
        print(f"\n🔗 Sharing document with {len(attendee_emails)} attendees...")
        
        if attendee_emails:
            for email in attendee_emails:
                try:
                    self.drive_service.permissions().create(
                        fileId=document_id,
                        body={
                            'type': 'user',
                            'role': 'reader',
                            'emailAddress': email
                        },
                        fields='id'
                    ).execute()
                except HttpError as e:
                    print(f"Failed to share with {email}: {e}")
        else:
            # As a last resort, make doc accessible to anyone with the link (unlisted)
            try:
                self.drive_service.permissions().create(
                    fileId=document_id,
                    body={
                        'type': 'anyone',
                        'role': 'reader'
                    },
                    fields='id'
                ).execute()
                print("⚠️  No attendee emails available – document set to Anyone with the link (read-only)")
            except HttpError as e:
                print(f"Failed to set public permission: {e}")

        print("✅ Document sharing step complete")
        
        # Get the shareable link
        file_info = self.drive_service.files().get(
            fileId=document_id,
            fields='webViewLink'
        ).execute()
        
        return file_info.get('webViewLink', '')
    
    def send_email_notifications(self, attendee_emails, meeting_title, doc_link):
        """Send email notifications to attendees."""
        print(f"\n📧 Sending email notifications...")
        
        subject = f"Meeting Notes Available: {meeting_title}"
        body = f"""
        Hello,
        
        The AI-generated meeting notes for "{meeting_title}" are now available.
        
        You can access the notes here: {doc_link}
        
        The document includes:
        - Executive Summary
        - Key Decisions
        - Action Items
        - Important Points
        
        Best regards,
        AI Meeting Assistant
        """
        
        for email in attendee_emails:
            try:
                message = {
                    'raw': self._create_message_raw(email, subject, body)
                }
                self.gmail_service.users().messages().send(
                    userId='me',
                    body=message
                ).execute()
                print(f"✅ Email sent to {email}")
            except HttpError as e:
                print(f"Failed to send email to {email}: {e}")
    
    def _create_message_raw(self, to, subject, body):
        """Create a base64-encoded email message."""
        import base64
        from email.mime.text import MIMEText
        
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return raw_message
    
    def run(self):
        """Execute the complete automation workflow."""
        print("🤖 Welcome to the AI Meeting Notes Assistant!")
        print("=" * 50)
        
        # Step 1: List recent meetings
        meetings = self.list_recent_meetings()
        if not meetings:
            print("No recent meetings found.")
            return
        
        # Step 2: Let user select a meeting
        selected_meeting = self.display_meetings_for_selection(meetings)
        meeting_title = selected_meeting.get('summary', 'Untitled Meeting')
        print(f"\n✅ Selected: {meeting_title}")
        
        # Step 3: Find conference record(s)
        conference_record = self.find_conference_record(selected_meeting)

        # If initial record found but no transcript later, we'll look at siblings.
        records_for_code = []
        meeting_code = None
        if 'conferenceData' in selected_meeting and selected_meeting['conferenceData'].get('conferenceId'):
            meeting_code = selected_meeting['conferenceData']['conferenceId']
        elif 'hangoutLink' in selected_meeting:
            meeting_code = selected_meeting['hangoutLink'].split('/')[-1]

        if meeting_code:
            records_for_code = self._list_conference_records(meeting_code)
        if conference_record and conference_record not in records_for_code:
            records_for_code.insert(0, conference_record)

        # Try to pick record with a ready transcript first
        chosen_record, transcript = self._pick_record_with_transcript(records_for_code)

        # If none ready, fall back to waiting on most recent record
        if not transcript:
            record_to_wait = conference_record or (records_for_code[-1] if records_for_code else None)
            if not record_to_wait:
                print("❌ Could not find conference record for this meeting.")
                return
            transcript = self.wait_for_transcript(record_to_wait['name'])
            if not transcript:
                print("❌ Transcript not available.")
                return
            chosen_record = record_to_wait
        
        # Step 5: Fetch transcript entries
        transcript_text = self.fetch_transcript_entries(transcript['name'])
        if not transcript_text:
            print("❌ Failed to fetch transcript content.")
            return
        
        # Step 6: Generate AI summary
        ai_summary = self.generate_ai_summary(transcript_text, meeting_title)
        
        # Step 7: Create Google Doc
        document_id = self.create_google_doc(meeting_title, ai_summary, selected_meeting)
        
        # Step 8: Share document with attendees
        attendee_emails = [att.get('email') for att in selected_meeting.get('attendees', []) if att.get('email')]

        # Fallback: include organizer if attendees list is empty
        if not attendee_emails:
            organizer_email = selected_meeting.get('organizer', {}).get('email')
            if organizer_email:
                attendee_emails.append(organizer_email)
        doc_link = self.share_document(document_id, attendee_emails)
        
        # Step 9: Send email notifications
        self.send_email_notifications(attendee_emails, meeting_title, doc_link)
        
        print(f"\n✨ Success! Meeting notes have been created and shared.")
        print(f"📄 Document link: {doc_link}")




def main():
    """Main entry point for the script."""
    print("🤖 AI-Powered Meeting Notes Automation")
    print("=" * 50)
    
    try:
        automation = MeetingNotesAutomation()
        automation.run()
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error: {e}")
        
        if "Invalid or missing credentials" in error_msg or "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in error_msg:
            print(f"\n❌ Authentication Error")
            print("\n🔧 To fix authentication issues:")
            print("1. Delete existing token file (if any):")
            print("   rm token.json")
            print("2. Run centralized authentication setup:")
            print("   python auth.py")
            print("3. Re-run this script")
            print("\nThe auth.py script will set up OAuth with all required scopes for:")
            print("  ✓ Google Meet API (conference records & transcripts)")
            print("  ✓ Google Calendar API")
            print("  ✓ Google Docs API") 
            print("  ✓ Google Drive API")
            print("  ✓ Gmail API")
            
        else:
            print(f"\n❌ Error: {e}")
            print("\nPlease check:")
            print("1. Your token.json file is in the root directory")
            print("2. GEMINI_API_KEY is set in .env file")
            print("3. Google Meet API is enabled in Cloud Console")
            print("4. Your token has the required OAuth scopes")
            print("5. Run 'python auth.py' to re-authenticate if needed")
            
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 