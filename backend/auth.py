"""
AgenticSuite OAuth Authentication Setup

This is the centralized authentication setup for the entire AgenticSuite project.
It configures OAuth2 credentials with all required scopes for:
- Email Agent (Gmail automation)
- Meeting Agent (Google Meet notes automation)
- Contract Agent (if using Google services)

Run this once to set up authentication for all agents.
"""

from __future__ import print_function
import os.path
import pickle
import google.auth
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Comprehensive OAuth2 scopes for all AgenticSuite agents
SCOPES = [
    # Calendar API (Email Agent & Meeting Agent)
    'https://www.googleapis.com/auth/calendar.readonly',        # Read calendar events
    'https://www.googleapis.com/auth/calendar',                 # Full calendar access
    
    # Gmail API (Email Agent & Meeting Agent notifications)
    'https://www.googleapis.com/auth/gmail.send',              # Send emails
    'https://www.googleapis.com/auth/gmail.readonly',          # Read emails
    'https://www.googleapis.com/auth/gmail.modify',            # Modify emails (archive, labels, etc.)
    
    # Google Meet API (Meeting Agent)
    'https://www.googleapis.com/auth/meetings.space.readonly', # Read Meet spaces & conference records
    'https://www.googleapis.com/auth/meetings.space.created',  # Access created Meet spaces
    
    # Google Docs API (Meeting Agent)
    'https://www.googleapis.com/auth/documents',               # Create/edit Google Docs
    
    # Google Drive API (Meeting Agent)
    'https://www.googleapis.com/auth/drive.file',              # Access specific Drive files
    'https://www.googleapis.com/auth/drive',                   # Full Drive access (fallback)
]

# Create organized directory structure for all agents
def create_agent_directories():
    """Create organized directories for all agents"""
    directories = [
        "mailDATAtemp/data", 
        "mailDATAtemp/logs",
        "meetingDATAtemp/data",
        "meetingDATAtemp/logs",
        "contractDATAtemp/data",
        "contractDATAtemp/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created organized directory structure for all agents")
    print("📄 Authentication uses centralized token.json in root directory")

def authenticate_google():
    """Authenticate and create tokens for all AgenticSuite agents"""
    print("🔐 AgenticSuite OAuth Authentication Setup")
    print("=" * 50)
    print("\nRequired scopes for all agents:")
    for scope in SCOPES:
        print(f"  ✓ {scope}")
    
    # Create organized directories first
    create_agent_directories()
    
    creds = None
    
    # Check for existing token
    if os.path.exists('token.json'):
        print("\n📂 Found existing token.json, loading...")
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # Validate and refresh credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired credentials...")
            try:
                creds.refresh(Request())
                print("✅ Credentials refreshed successfully")
            except Exception as e:
                print(f"❌ Failed to refresh credentials: {e}")
                creds = None
        
        # Run OAuth flow if needed
        if not creds:
            print("\n🌐 Starting OAuth authentication flow...")
            
            # Look for client secret file
            possible_files = [
                'client_secret_815841228751-t95l5cbijftl9b3euol7lkjnj1ne2d6t.apps.googleusercontent.com.json',
                'client_secret.json',
                'credentials.json'
            ]
            
            credentials_file = None
            for file_path in possible_files:
                if os.path.exists(file_path):
                    credentials_file = file_path
                    break
            
            if not credentials_file:
                print("\n❌ Error: No OAuth client secret file found!")
                print("\nSearched for:")
                for file_path in possible_files:
                    print(f"  - {file_path}")
                print("\nPlease:")
                print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
                print("2. Enable these APIs:")
                print("   - Google Meet API")
                print("   - Google Calendar API") 
                print("   - Google Docs API")
                print("   - Google Drive API")
                print("   - Gmail API")
                print("3. Create OAuth2 credentials (Desktop Application)")
                print("4. Download as one of the expected filenames above")
                raise FileNotFoundError("OAuth client secret file not found")
            
            print(f"✅ Found credentials file: {credentials_file}")
            print("🌐 Opening browser for authentication...")
            print("Please sign in and grant permissions for all requested scopes.")
            
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
            print("✅ OAuth authentication successful!")
    
    # Save credentials to main location and all agent directories
    save_credentials_to_all_agents(creds)
    
    return creds

def save_credentials_to_all_agents(creds):
    """Save credentials to centralized token.json"""
    print("\n📁 Saving credentials to centralized location...")
    
    # Save to single root token location
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    print("✅ Saved to: token.json")
    
    print("\n🎉 Authentication setup complete for all AgenticSuite agents!")
    print("\nAll agents will use the centralized token.json file.")
    print("\nYou can now run:")
    print("  - Email Agent: python email_agent.py")
    print("  - Meeting Agent: python meeting_notes_automation.py") 
    print("  - Contract Agent: python contract_agent.py")

# Example usage: Test Google Drive API
if __name__ == '__main__':
    creds = authenticate_google()
    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(pageSize=10).execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
    else:
        print('Files:')
        for item in items:
            print(f"{item['name']} ({item['id']})")
