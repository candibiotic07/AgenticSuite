#!/usr/bin/env python3
"""Test script to verify the meeting notes automation setup."""

import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def test_google_auth():
    """Test Google authentication."""
    print("🔍 Testing Google Authentication...")
    
    try:
        if not os.path.exists('token.json'):
            print("❌ token.json not found! Run auth.py first.")
            return False
            
        creds = Credentials.from_authorized_user_file('token.json')
        print("✅ Successfully loaded credentials from token.json")
        
        # Test Calendar API
        print("\n📅 Testing Calendar API access...")
        calendar_service = build('calendar', 'v3', credentials=creds)
        calendar_list = calendar_service.calendarList().list(maxResults=1).execute()
        print("✅ Calendar API is working!")
        
        # Test Drive API
        print("\n📁 Testing Drive API access...")
        drive_service = build('drive', 'v3', credentials=creds)
        files = drive_service.files().list(pageSize=1).execute()
        print("✅ Drive API is working!")
        
        # Test Gmail API
        print("\n📧 Testing Gmail API access...")
        gmail_service = build('gmail', 'v1', credentials=creds)
        labels = gmail_service.users().labels().list(userId='me').execute()
        print("✅ Gmail API is working!")
        
        return True
        
    except HttpError as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gemini_api():
    """Test Gemini API setup."""
    print("\n🤖 Testing Gemini API...")
    
    # Try to load from .env file first
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️  python-dotenv not installed, checking system environment variables only")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("   Set it in .env file: GEMINI_API_KEY=your-api-key-here")
        print("   Or set environment variable: $env:GEMINI_API_KEY = 'your-api-key-here'")
        return False
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say 'Hello, Meeting Notes!'")
        print("✅ Gemini API is working!")
        print(f"   Response: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Meeting Notes Automation Setup Test")
    print("=" * 50)
    
    # Check for required files
    print("\n📄 Checking required files...")
    required_files = ['token.json', 'meeting_notes_automation.py', 'auth.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
    
    # Test Google APIs
    google_ok = test_google_auth()
    
    # Test Gemini API
    gemini_ok = test_gemini_api()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary:")
    if google_ok and gemini_ok:
        print("✅ All systems go! You're ready to run meeting_notes_automation.py")
    else:
        print("❌ Some issues need to be resolved before running the automation.")
        if not google_ok:
            print("   - Fix Google API authentication issues")
        if not gemini_ok:
            print("   - Set up Gemini API key")

if __name__ == "__main__":
    main() 