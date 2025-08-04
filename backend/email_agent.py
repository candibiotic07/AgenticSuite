"""
AI-Powered Email Automation Agent

This agent stores data in organized directories:
- mailDATAtemp/
  ├── data/          # Metrics and CSV files
  └── logs/          # Log files
- token.json         # Centralized authentication (root directory)

The agent will automatically migrate existing files to the new structure.
"""

import os
import json
import datetime
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import csv
from dotenv import load_dotenv

# Google API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Create main data directory structure
BASE_DATA_DIR = "mailDATAtemp"
os.makedirs(os.path.join(BASE_DATA_DIR, "logs"), exist_ok=True)    # Log files
os.makedirs(os.path.join(BASE_DATA_DIR, "data"), exist_ok=True)    # Metrics and data files
# Note: Authentication uses centralized token.json in root directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DATA_DIR, "logs", "email_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Email categories
class EmailCategory(Enum):
    MEETING_REQUEST = "MEETING_REQUEST"
    INVOICE = "INVOICE"
    SCHEDULING = "SCHEDULING"
    FAQ = "FAQ"
    NEWSLETTER = "NEWSLETTER"
    SPAM = "SPAM"
    OTHER = "OTHER"

@dataclass
class Email:
    """Email data structure"""
    message_id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    timestamp: datetime.datetime

@dataclass
class TimeSlot:
    """Available time slot for meetings"""
    start: datetime.datetime
    end: datetime.datetime
    
    def __str__(self):
        # Format with IST timezone indicator
        return f"{self.start.strftime('%a, %b %d · %I:%M %p')} - {self.end.strftime('%I:%M %p')} IST"

@dataclass
class ClassificationResult:
    """Email classification result"""
    category: EmailCategory
    confidence: float

class EmailAgent:
    """AI Email Automation Agent"""
    
    # Google API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/calendar',
        'https://www.googleapis.com/auth/calendar.readonly'
    ]
    
    # Email templates
    TEMPLATES = {
        EmailCategory.INVOICE: """Hi {sender_name},

Thank you for sending the invoice {invoice_ref}. I've received it and will process it according to our standard procedures.

Best regards""",
        
        EmailCategory.FAQ: """Hi {sender_name},

Thank you for your question. {answer}

If you need any further clarification, please don't hesitate to ask.

Best regards""",
        
        EmailCategory.SCHEDULING: """Hi {sender_name},

Thank you for your response regarding the meeting. {details}

Best regards"""
    }
    
    def __init__(self, lookback_hours: int = 72, confidence_threshold: float = 0.7):
        """Initialize the Email Agent
        
        Args:
            lookback_hours: Hours to look back for emails
            confidence_threshold: Minimum confidence for classification
        """
        self.lookback_hours = lookback_hours
        self.confidence_threshold = confidence_threshold
        self.creds = self._authenticate()
        
        # Initialize Google services
        self.gmail_service = build('gmail', 'v1', credentials=self.creds)
        self.calendar_service = build('calendar', 'v3', credentials=self.creds)
        
        # Initialize Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize metrics tracking
        self.metrics_file = os.path.join(BASE_DATA_DIR, "data", "email_agent_metrics.csv")
        self._init_metrics_file()
        
        # Check for and migrate existing files
        self._migrate_existing_files()
        
        logger.info("Email Agent initialized successfully")
        logger.info(f"Data directory: {BASE_DATA_DIR}")
        logger.info(f"  - Logs: {os.path.join(BASE_DATA_DIR, 'logs')}")
        logger.info(f"  - Data: {os.path.join(BASE_DATA_DIR, 'data')}")  
        logger.info("Authentication: Centralized token.json in root directory")
    
    def _authenticate(self) -> Credentials:
        """Authenticate using centralized credentials"""
        creds = None
        token_file = "token.json"  # Use centralized token in root directory
        
        # Load existing token
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        
        # Refresh token if expired
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save the refreshed credentials
            with open(token_file, 'w') as token:
                token.write(creds.to_json())
        
        if not creds or not creds.valid:
            raise Exception(f"Invalid credentials. Please run 'python auth.py' to set up authentication. Token should be in {token_file}")
        
        return creds
    
    def _migrate_existing_files(self):
        """Migrate existing files to new directory structure"""
        # Note: token.json stays in root directory for centralized access
        
        # Migrate metrics file
        old_metrics_path = "email_agent_metrics.csv"
        new_metrics_path = os.path.join(BASE_DATA_DIR, "data", "email_agent_metrics.csv")
        
        if os.path.exists(old_metrics_path) and not os.path.exists(new_metrics_path):
            try:
                import shutil
                shutil.move(old_metrics_path, new_metrics_path)
                logger.info(f"Migrated metrics file to {new_metrics_path}")
                print(f"✅ Migrated metrics file to {new_metrics_path}")
            except Exception as e:
                logger.warning(f"Could not migrate metrics file: {e}")
        
        # Migrate log file
        old_log_path = "email_agent.log"
        new_log_path = os.path.join(BASE_DATA_DIR, "logs", "email_agent.log")
        
        if os.path.exists(old_log_path) and not os.path.exists(new_log_path):
            try:
                import shutil
                shutil.move(old_log_path, new_log_path)
                logger.info(f"Migrated log file to {new_log_path}")
                print(f"✅ Migrated log file to {new_log_path}")
            except Exception as e:
                logger.warning(f"Could not migrate log file: {e}")
    
    def _init_metrics_file(self):
        """Initialize metrics CSV file"""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'email_id', 'category', 'action_taken',
                    'draft_accepted', 'meeting_scheduled', 'user_edited'
                ])
    
    def fetch_recent_emails(self) -> List[Email]:
        """Fetch emails from the past lookback period"""
        try:
            # Calculate cutoff time
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=self.lookback_hours)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            logger.info(f"Fetching emails after {cutoff_time}")
            
            # Query for messages
            query = f'after:{cutoff_timestamp} -from:me'
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=50
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                # Get full message
                message = self.gmail_service.users().messages().get(
                    userId='me',
                    id=msg['id']
                ).execute()
                
                # Extract email data
                email = self._parse_email(message)
                if email:
                    emails.append(email)
            
            logger.info(f"Fetched {len(emails)} emails")
            return emails
            
        except HttpError as error:
            logger.error(f"Error fetching emails: {error}")
            return []
    
    def _parse_email(self, message: Dict) -> Optional[Email]:
        """Parse Gmail message into Email object"""
        try:
            headers = message['payload'].get('headers', [])
            
            # Extract headers
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), '')
            date_str = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            # Parse timestamp
            timestamp = datetime.datetime.now()  # Default
            if date_str:
                try:
                    # Simple parsing - might need more robust solution
                    timestamp = datetime.datetime.strptime(date_str[:31], '%a, %d %b %Y %H:%M:%S')
                except:
                    pass
            
            # Extract body
            body = self._get_email_body(message['payload'])
            
            return Email(
                message_id=message['id'],
                thread_id=message['threadId'],
                sender=sender,
                subject=subject,
                body=body,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error parsing email: {e}")
            return None
    
    def _get_email_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body = self._decode_base64(data)
                    break
        elif payload['body'].get('data'):
            body = self._decode_base64(payload['body']['data'])
        
        return body
    
    def _decode_base64(self, data: str) -> str:
        """Decode base64 email data"""
        import base64
        return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    
    def classify_email(self, email: Email) -> ClassificationResult:
        """Classify email using Gemini"""
        prompt = f"""Analyze the following email and classify it into one of the following categories:

MEETING_REQUEST: The sender wants to schedule a new meeting, call, or appointment.
INVOICE: The email contains a bill, receipt, or a request for payment.
SCHEDULING: This is a reply to a previous meeting request or a calendar invite update.
FAQ: The sender is asking a common question that can be answered with a template.
NEWSLETTER: Automated promotional or informational email.
SPAM: Unsolicited or irrelevant commercial email.
OTHER: Any other type of email.

Subject: {email.subject}
From: {email.sender}
Body: {email.body[:1000]}  # Limit body length

Respond with only the category name and a confidence score (0-1) separated by a comma.
Example: MEETING_REQUEST,0.95"""

        try:
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip().split(',')
            
            category = EmailCategory[result[0].strip()]
            confidence = float(result[1].strip())
            
            logger.info(f"Classified email from {email.sender} as {category.value} (confidence: {confidence})")
            
            return ClassificationResult(category=category, confidence=confidence)
            
        except Exception as e:
            logger.error(f"Error classifying email: {e}")
            return ClassificationResult(category=EmailCategory.OTHER, confidence=0.0)
    
    def handle_meeting_request(self, email: Email) -> List[TimeSlot]:
        """Handle meeting request and find available slots"""
        # Parse meeting details
        meeting_details = self._parse_meeting_details(email)
        duration_minutes = meeting_details.get('duration', 30)
        
        # Get available slots
        available_slots = self._find_available_slots(
            duration_minutes=duration_minutes,
            days_ahead=14
        )
        
        # Return top 3 slots
        return available_slots[:3]
    
    def _parse_meeting_details(self, email: Email) -> Dict:
        """Parse meeting details from email using LLM"""
        prompt = f"""Extract meeting details from this email:

Subject: {email.subject}
Body: {email.body}

Extract:
1. Meeting duration in minutes (default 30 if not specified)
2. Any specific date/time preferences
3. Meeting type (call, in-person, video)

Respond in JSON format:
{{"duration": 30, "preferences": "next week", "type": "video"}}"""

        try:
            response = self.gemini_model.generate_content(prompt)
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {"duration": 30, "preferences": "", "type": "video"}
    
    def _find_available_slots(self, duration_minutes: int, days_ahead: int) -> List[TimeSlot]:
        """Find available time slots in calendar"""
        import pytz
        
        # Set timezone to IST
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
        
        # Calculate time range in UTC for API
        time_min = now.replace(hour=0, minute=0, second=0, microsecond=0)
        time_max = time_min + datetime.timedelta(days=days_ahead)
        
        # Convert to UTC for API
        time_min_utc = time_min.astimezone(pytz.UTC).isoformat().replace('+00:00', 'Z')
        time_max_utc = time_max.astimezone(pytz.UTC).isoformat().replace('+00:00', 'Z')
        
        try:
            # Get busy times from calendar
            body = {
                "timeMin": time_min_utc,
                "timeMax": time_max_utc,
                "timeZone": "Asia/Kolkata",
                "items": [{"id": "primary"}]
            }
            
            freebusy_result = self.calendar_service.freebusy().query(body=body).execute()
            busy_times = freebusy_result['calendars']['primary']['busy']
            
            # Convert busy times to IST datetime objects
            busy_periods = []
            for busy in busy_times:
                busy_start = datetime.datetime.fromisoformat(busy['start'].replace('Z', '+00:00'))
                busy_end = datetime.datetime.fromisoformat(busy['end'].replace('Z', '+00:00'))
                # Convert to IST
                busy_start = busy_start.astimezone(ist)
                busy_end = busy_end.astimezone(ist)
                busy_periods.append((busy_start, busy_end))
            
            # Find free slots
            available_slots = []
            
            # Start from tomorrow if current time is past 5 PM
            current_date = now.date()
            if now.hour >= 17:
                current_date += datetime.timedelta(days=1)
            
            slots_found = 0
            days_checked = 0
            
            while slots_found < 10 and days_checked < days_ahead:
                # Set working hours for the day (9 AM to 6 PM IST)
                day_start = ist.localize(datetime.datetime.combine(
                    current_date, 
                    datetime.time(9, 0)
                ))
                day_end = ist.localize(datetime.datetime.combine(
                    current_date, 
                    datetime.time(18, 0)
                ))
                
                # Skip weekends
                if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    current_date += datetime.timedelta(days=1)
                    days_checked += 1
                    continue
                
                # Check slots every 30 minutes
                current_time = day_start
                
                # If it's today, start from next available 30-min slot
                if current_date == now.date():
                    # Round up to next 30-min slot
                    minutes = now.minute
                    if minutes < 30:
                        next_slot = now.replace(minute=30, second=0, microsecond=0)
                    else:
                        next_slot = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                    
                    if next_slot > current_time and next_slot < day_end:
                        current_time = next_slot
                
                while current_time + datetime.timedelta(minutes=duration_minutes) <= day_end:
                    slot_end = current_time + datetime.timedelta(minutes=duration_minutes)
                    
                    # Check if this slot conflicts with any busy period
                    is_free = True
                    for busy_start, busy_end in busy_periods:
                        # Check for overlap
                        if not (slot_end <= busy_start or current_time >= busy_end):
                            is_free = False
                            break
                    
                    # Also check if slot is in the past
                    if current_time <= now:
                        is_free = False
                    
                    if is_free:
                        available_slots.append(TimeSlot(start=current_time, end=slot_end))
                        slots_found += 1
                        
                        if slots_found >= 10:  # Limit to 10 slots
                            break
                    
                    # Move to next 30-min slot
                    current_time += datetime.timedelta(minutes=30)
                
                # Move to next day
                current_date += datetime.timedelta(days=1)
                days_checked += 1
            
            return available_slots
            
        except Exception as e:
            logger.error(f"Error finding available slots: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def create_calendar_event(self, email: Email, slot: TimeSlot, meeting_details: Dict) -> str:
        """Create calendar event with Google Meet link"""
        try:
            # Extract sender name and email
            sender_name = email.sender.split('<')[0].strip()
            sender_email = email.sender.split('<')[-1].strip('>')
            
            event = {
                'summary': f'Meeting with {sender_name}',
                'description': f'Meeting scheduled from email: {email.subject}',
                'start': {
                    'dateTime': slot.start.isoformat(),
                    'timeZone': 'Asia/Kolkata',
                },
                'end': {
                    'dateTime': slot.end.isoformat(),
                    'timeZone': 'Asia/Kolkata',
                },
                'attendees': [
                    {'email': sender_email},
                ],
                'conferenceData': {
                    'createRequest': {
                        'requestId': f"meet-{email.message_id[:20]}-{int(time.time())}",
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'reminders': {
                    'useDefault': True,
                },
            }
            
            # Create event
            event_result = self.calendar_service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            
            logger.info(f"Created calendar event: {event_result.get('htmlLink')}")
            
            # Force calendar service refresh to ensure subsequent calls see this event
            self.calendar_service = build('calendar', 'v3', credentials=self.creds)
            
            return event_result.get('id', '')
            
        except Exception as e:
            logger.error(f"Error creating calendar event: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def generate_meeting_confirmation(self, email: Email, slot: TimeSlot) -> str:
        """Generate meeting confirmation email"""
        sender_name = email.sender.split('<')[0].strip() or "there"
        
        confirmation = f"""Hi {sender_name},

Thanks for reaching out. I've gone ahead and scheduled our meeting for {slot.start.strftime('%a, %b %d at %I:%M %p')} IST.

You should have received a Google Calendar invitation with a Meet link shortly.

Looking forward to it!

Best regards"""
        
        return confirmation
    
    def generate_template_response(self, email: Email, category: EmailCategory) -> str:
        """Generate template-based response"""
        if category not in self.TEMPLATES:
            return ""
        
        template = self.TEMPLATES[category]
        
        # Use LLM to fill template
        prompt = f"""Fill in this email template based on the original email:

Original Email:
From: {email.sender}
Subject: {email.subject}
Body: {email.body[:500]}

Template to fill:
{template}

Provide a natural, complete response. Keep placeholders filled appropriately."""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating template response: {e}")
            return ""
    
    def send_email(self, to: str, subject: str, body: str, thread_id: Optional[str] = None) -> bool:
        """Send email via Gmail API"""
        try:
            message = {
                'raw': self._create_message(to, subject, body, thread_id)
            }
            
            sent = self.gmail_service.users().messages().send(
                userId='me',
                body=message
            ).execute()
            
            logger.info(f"Email sent successfully: {sent['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _create_message(self, to: str, subject: str, body: str, thread_id: Optional[str] = None) -> str:
        """Create email message"""
        import base64
        from email.mime.text import MIMEText
        
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject
        
        if thread_id:
            message['References'] = thread_id
            message['In-Reply-To'] = thread_id
        
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return raw
    
    def archive_email(self, email_id: str) -> bool:
        """Archive an email by removing it from inbox"""
        try:
            # Remove the INBOX label to archive the email
            modify_request = {
                'removeLabelIds': ['INBOX']
            }
            
            result = self.gmail_service.users().messages().modify(
                userId='me',
                id=email_id,
                body=modify_request
            ).execute()
            
            logger.info(f"Email {email_id} archived successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving email {email_id}: {e}")
            return False
    
    def log_action(self, email_id: str, category: str, action: str, 
                   draft_accepted: bool, meeting_scheduled: bool, user_edited: bool):
        """Log user action for metrics"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().isoformat(),
                email_id,
                category,
                action,
                draft_accepted,
                meeting_scheduled,
                user_edited
            ])
    
    def get_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        metrics = {
            'emails_processed': 0,
            'meetings_scheduled': 0,
            'drafts_accepted': 0,
            'automation_rate': 0.0,
            'classification_accuracy': 0.0
        }
        
        try:
            with open(self.metrics_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                metrics['emails_processed'] = len(rows)
                metrics['meetings_scheduled'] = sum(1 for r in rows if r['meeting_scheduled'] == 'True')
                metrics['drafts_accepted'] = sum(1 for r in rows if r['draft_accepted'] == 'True')
                
                if metrics['emails_processed'] > 0:
                    metrics['automation_rate'] = metrics['drafts_accepted'] / metrics['emails_processed']
        except:
            pass
        
        return metrics
    
    def process_emails(self):
        """Main processing loop"""
        print("\n🤖 Email Agent Started\n" + "="*50)
        
        # Fetch recent emails
        emails = self.fetch_recent_emails()
        
        if not emails:
            print("No new emails to process.")
            return
        
        print(f"\nFound {len(emails)} emails to process\n")
        
        for i, email in enumerate(emails, 1):
            print(f"\n[{i}/{len(emails)}] Processing email from: {email.sender}")
            print(f"Subject: {email.subject}")
            print("-" * 50)
            
            # Classify email
            classification = self.classify_email(email)
            
            if classification.confidence < self.confidence_threshold:
                print(f"⚠️  Low confidence classification ({classification.confidence:.2f}). Skipping...")
                self.log_action(email.message_id, classification.category.value, 
                              "skipped", False, False, False)
                continue
            
            print(f"📧 Classified as: {classification.category.value} (confidence: {classification.confidence:.2f})")
            
            # Handle based on category
            if classification.category == EmailCategory.MEETING_REQUEST:
                self._handle_meeting_request_interactive(email)
            
            elif classification.category in [EmailCategory.INVOICE, EmailCategory.FAQ, EmailCategory.SCHEDULING]:
                self._handle_template_response_interactive(email, classification.category)
            
            else:
                print(f"ℹ️  Category {classification.category.value} - No action needed")
                self.log_action(email.message_id, classification.category.value, 
                              "no_action", False, False, False)
        
        # Show metrics
        print("\n" + "="*50)
        print("📊 Session Metrics:")
        metrics = self.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    def _handle_meeting_request_interactive(self, email: Email):
        """Interactively handle meeting request"""
        print("\n🗓️  This is a MEETING REQUEST. Finding available slots...")
        
        # Find available slots
        slots = self.handle_meeting_request(email)
        
        if not slots:
            print("❌ No available slots found")
            self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                          "no_slots", False, False, False)
            return
        
        # Present options
        print("\nAvailable slots:")
        for i, slot in enumerate(slots, 1):
            print(f"  [{i}] {slot}")
        
        # Get user choice
        while True:
            choice = input("\nChoose a slot [1-3] or [s]kip: ").strip().lower()
            
            if choice == 's':
                print("Skipped")
                self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                              "skipped", False, False, False)
                return
            
            try:
                slot_idx = int(choice) - 1
                if 0 <= slot_idx < len(slots):
                    selected_slot = slots[slot_idx]
                    break
            except:
                pass
            
            print("Invalid choice. Please try again.")
        
        # Create calendar event
        print("\n📅 Creating calendar event...")
        meeting_details = self._parse_meeting_details(email)
        event_id = self.create_calendar_event(email, selected_slot, meeting_details)
        
        if not event_id:
            print("❌ Failed to create calendar event")
            self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                          "event_failed", False, False, False)
            return
        
        # Generate confirmation
        confirmation = self.generate_meeting_confirmation(email, selected_slot)
        print("\n📝 Draft confirmation email:")
        print("-" * 40)
        print(confirmation)
        print("-" * 40)
        
        # Get approval
        action = input("\n[s]end, [e]dit, or [c]ancel? ").strip().lower()
        
        if action == 'e':
            print("\nEnter your edited message (type 'END' on a new line when done):")
            print("Current message:")
            print("-" * 40)
            print(confirmation)
            print("-" * 40)
            print("\nYour edited version:")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                except KeyboardInterrupt:
                    print("\nEdit cancelled")
                    return
            confirmation = '\n'.join(lines)
            edited = True
        else:
            edited = False
        
        if action in ['s', 'e']:
            # Send confirmation
            sender_email = email.sender.split('<')[-1].strip('>')
            if self.send_email(sender_email, f"Re: {email.subject}", 
                             confirmation, email.thread_id):
                print("✅ Confirmation sent!")
                self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                              "sent", not edited, True, edited)
            else:
                print("❌ Failed to send confirmation")
                self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                              "send_failed", False, True, edited)
        else:
            print("Cancelled")
            self.log_action(email.message_id, EmailCategory.MEETING_REQUEST.value, 
                          "cancelled", False, True, False)
    
    def _handle_template_response_interactive(self, email: Email, category: EmailCategory):
        """Interactively handle template-based responses"""
        print(f"\n📝 Generating {category.value} response...")
        
        # Generate response
        response = self.generate_template_response(email, category)
        
        if not response:
            print("❌ Failed to generate response")
            self.log_action(email.message_id, category.value, "generation_failed", 
                          False, False, False)
            return
        
        print("\nDraft response:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Get approval
        action = input("\n[s]end, [e]dit, or [c]ancel? ").strip().lower()
        
        if action == 'e':
            print("\nEnter your edited message (type 'END' on a new line when done):")
            print("Current message:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            print("\nYour edited version:")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    lines.append(line)
                except KeyboardInterrupt:
                    print("\nEdit cancelled")
                    return
            response = '\n'.join(lines)
            edited = True
        else:
            edited = False
        
        if action in ['s', 'e']:
            # Send response
            sender_email = email.sender.split('<')[-1].strip('>')
            if self.send_email(sender_email, f"Re: {email.subject}", 
                             response, email.thread_id):
                print("✅ Response sent!")
                self.log_action(email.message_id, category.value, "sent", 
                              not edited, False, edited)
            else:
                print("❌ Failed to send response")
                self.log_action(email.message_id, category.value, "send_failed", 
                              False, False, edited)
        else:
            print("Cancelled")
            self.log_action(email.message_id, category.value, "cancelled", 
                          False, False, False)


def main():
    """Main entry point"""
    try:
        # Initialize agent
        agent = EmailAgent(lookback_hours=72)
        
        # Process emails
        agent.process_emails()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. Your token.json file is in the root directory")
        print("2. GEMINI_API_KEY is set in .env file")
        print("3. You have proper Google API permissions")
        print("4. Run 'python auth.py' if you need to set up authentication")


if __name__ == "__main__":
    main() 