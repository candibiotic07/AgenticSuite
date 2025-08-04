# 🛠️ AgenticSuite Setup Guide

**Complete installation and configuration guide for AgenticSuite - from zero to automation in 30 minutes.**

---

## 📋 **Prerequisites**

Before starting, ensure you have:

- **Python 3.8 or newer** ([Download here](https://python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/downloads/))
- **Google Account** with access to Gmail, Calendar, and Drive
- **Google Cloud Platform Account** (free tier is sufficient)
- **Modern web browser** (Chrome, Firefox, Safari, or Edge)

---

## 🚀 **Step 1: Clone and Setup Environment**

### **1.1 Clone the Repository**

```bash
# Clone the AgenticSuite repository
git clone https://github.com/yourusername/agenticsuite.git
cd agenticsuite

# Verify you're in the correct directory
ls -la
# You should see: backend/, frontend/, README.md, SETUP.md, requirements.txt
```

### **1.2 Create Virtual Environment**

**On macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show path to venv)
which python
```

**On Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show path to venv)
where python
```

### **1.3 Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation (should show 50+ packages)
pip list | wc -l
```

**If you encounter installation issues:**
```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output to diagnose issues
pip install -r requirements.txt -v
```

---

## 🔐 **Step 2: Google Cloud Platform Setup**

### **2.1 Create Google Cloud Project**

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **"New Project"** or use the project dropdown
3. Enter project name: `agenticsuite-automation`
4. Click **"Create"**
5. Wait for project creation (usually 30-60 seconds)

### **2.2 Enable Required APIs**

Navigate to **"APIs & Services" > "Library"** and enable these APIs:

```
✅ Gmail API
✅ Google Calendar API  
✅ Google Meet API
✅ Google Docs API
✅ Google Drive API
```

**Quick enable via search:**
- Search for each API name
- Click on the API result
- Click **"Enable"** button
- Repeat for all 5 APIs

### **2.3 Configure OAuth Consent Screen**

1. Go to **"APIs & Services" > "OAuth consent screen"**
2. Choose **"External"** user type
3. Fill in the required information:
   - **App name**: `AgenticSuite`
   - **User support email**: Your email address
   - **Developer contact**: Your email address
4. Click **"Save and Continue"**

5. **Add OAuth Scopes** (click "Add or Remove Scopes"):
   ```
   https://www.googleapis.com/auth/calendar.readonly
   https://www.googleapis.com/auth/calendar
   https://www.googleapis.com/auth/gmail.send
   https://www.googleapis.com/auth/gmail.readonly
   https://www.googleapis.com/auth/gmail.modify
   https://www.googleapis.com/auth/meetings.space.readonly
   https://www.googleapis.com/auth/meetings.space.created
   https://www.googleapis.com/auth/documents
   https://www.googleapis.com/auth/drive.file
   https://www.googleapis.com/auth/drive
   ```

6. **Add Test Users** (click "Add Users"):
   - Add your Gmail address that you'll use with AgenticSuite

7. Click **"Save and Continue"** through remaining steps

### **2.4 Create OAuth2 Credentials**

1. Go to **"APIs & Services" > "Credentials"**
2. Click **"+ Create Credentials" > "OAuth client ID"**
3. Choose **"Desktop application"** as application type
4. Name: `AgenticSuite Desktop Client`
5. Click **"Create"**
6. **Download the JSON file** when prompted
7. **Important**: Rename the downloaded file to match one of these names:
   ```
   client_secret_815841228751-t95l5cbijftl9b3euol7lkjnj1ne2d6t.apps.googleusercontent.com.json
   OR
   client_secret.json
   OR 
   credentials.json
   ```
8. Move this file to the `backend/` directory

---

## 🔑 **Step 3: Setup Gemini AI API**

### **3.1 Get Gemini API Key**

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** 
4. Create a new API key in your Google Cloud project
5. **Copy the API key** (starts with `AIza...`)

### **3.2 Configure Environment Variables**

**Option A: Create .env file (Recommended)**
```bash
# Navigate to backend directory
cd backend

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Replace 'your_api_key_here' with your actual API key
# Example: GEMINI_API_KEY=AIzaSyC1234567890abcdef
```

**Option B: Set environment variable**
```bash
# On macOS/Linux
export GEMINI_API_KEY='your_api_key_here'

# On Windows
set GEMINI_API_KEY=your_api_key_here
```

---

## 🔐 **Step 4: Authentication Setup**

### **4.1 Run OAuth Authentication**

```bash
# Make sure you're in the backend directory
cd backend

# Run the authentication setup
python auth.py
```

### **4.2 Complete OAuth Flow**

The script will:

1. **Display required scopes** for verification
2. **Create agent directories** for data storage
3. **Open your web browser** automatically
4. **Guide you through Google sign-in**

**Follow these steps in the browser:**

1. **Sign in** with the Google account you added as a test user
2. **You may see "Google hasn't verified this app"** - this is expected
   - Click **"Advanced"**
   - Click **"Go to AgenticSuite (unsafe)"**
3. **Grant permissions** for all requested scopes:
   - Gmail access
   - Calendar access
   - Drive access
   - Meet access
   - Documents access
4. **Click "Allow"** to complete authorization

### **4.3 Verify Authentication**

After successful authentication, you should see:

```
✅ Created organized directory structure for all agents
📄 Authentication uses centralized token.json in root directory
✅ OAuth authentication successful!
✅ Saved to: token.json

🎉 Authentication setup complete for all AgenticSuite agents!

All agents will use the centralized token.json file.

You can now run:
  - Email Agent: python email_agent.py
  - Meeting Agent: python meeting_notes_automation.py
  - Contract Agent: python contract_agent.py
```

**Verify token.json was created:**
```bash
ls -la token.json
# Should show the token file with recent timestamp
```

---

## 🌐 **Step 5: Launch AgenticSuite Web Interface**

### **5.1 Start the Application**

```bash
# Make sure you're in the backend directory
cd backend

# Start the Flask web application
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: off
 * Press CTRL+C to quit
```

### **5.2 Access the Dashboard**

1. **Open your web browser**
2. **Navigate to**: `http://localhost:5000`
3. **You should see** the AgenticSuite dashboard

### **5.3 Test Agent Connectivity**

The web interface will automatically test connections to:
- ✅ Google Authentication (token.json)
- ✅ Gmail API access
- ✅ Calendar API access  
- ✅ Gemini AI API access

---

## 🎯 **Step 6: First Run - Test Each Agent**

### **6.1 Test Email Agent**

1. Click **"Email Agent"** in the dashboard
2. The agent will:
   - Fetch recent emails (last 72 hours)
   - Classify them using AI
   - Present actionable emails for review
3. **First run**: Approve or edit AI-generated responses
4. **Review metrics** after processing

### **6.2 Test Meeting Agent**

1. Click **"Meeting Agent"** in the dashboard
2. **Prerequisites**: You need a recent Google Meet recording
3. The agent will:
   - List recent meetings with recordings
   - Process transcripts when available
   - Generate professional meeting notes
   - Share with attendees automatically

### **6.3 Test Contract Agent**

1. Click **"Contract Agent"** in the dashboard
2. **Upload a sample contract** (PDF, DOCX, or TXT)
3. The agent will:
   - Extract and analyze clauses
   - Identify potential risks
   - Generate a detailed report
   - Learn from your feedback

---

## 🔧 **Troubleshooting Common Issues**

### **Problem: "No module named 'xyz'"**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### **Problem: "OAuth client secret file not found"**
```bash
# Verify file is in backend/ directory with correct name
ls backend/client_secret*.json
ls backend/credentials.json

# If missing, re-download from Google Cloud Console
```

### **Problem: "Failed to refresh credentials"**
```bash
# Delete existing token and re-authenticate
rm backend/token.json
cd backend
python auth.py
```

### **Problem: "Gemini API key not found"**
```bash
# Check if .env file exists
cat backend/.env

# If missing, create it:
echo "GEMINI_API_KEY=your_actual_api_key" > backend/.env
```

### **Problem: Flask app won't start**
```bash
# Check if port 5000 is in use
lsof -i :5000  # macOS/Linux
netstat -an | findstr :5000  # Windows

# Use different port if needed
export FLASK_RUN_PORT=5001
python app.py
```

### **Problem: Google API quota exceeded**
```bash
# Check API usage in Google Cloud Console
# Go to "APIs & Services" > "Quotas"
# Increase quotas if needed or wait for reset
```

---

## ⚙️ **Advanced Configuration**

### **Custom Agent Settings**

Edit these files to customize behavior:

```bash
# Email Agent Configuration
backend/email_agent.py
# Look for: EmailAgent(lookback_hours=72, confidence_threshold=0.7)

# Meeting Agent Configuration  
backend/meeting_agent.py
# Look for: MeetingNotesAutomation(max_wait_minutes=30)

# Contract Agent Configuration
backend/contract_agent.py
# Look for: ContractAgent(persist_directory='./contractDATAtemp')
```

### **Data Storage Locations**

AgenticSuite creates organized data directories: 

```
backend/
├── mailDATAtemp/           # Email agent data
│   ├── data/               # Metrics and preferences
│   └── logs/               # Processing logs
├── meetingDATAtemp/        # Meeting agent data
│   ├── data/               # Meeting records
│   └── logs/               # Processing logs
└── contractDATAtemp/       # Contract agent data
    ├── data/               # Risk patterns and feedback
    ├── logs/               # Analysis logs
    ├── reports/            # Generated reports
    ├── uploads/            # Uploaded contracts
    └── vectorstore/        # AI vector database
```

### **Security Best Practices**

1. **Never commit token.json** to version control
2. **Keep API keys secure** in .env files only
3. **Regularly rotate** OAuth tokens and API keys
4. **Monitor API usage** in Google Cloud Console
5. **Use strong passwords** for your Google account

---

## 🎉 **Congratulations!**

You've successfully set up AgenticSuite! Your intelligent automation platform is now ready to:

- **📧 Automate email workflows** with smart classification and responses
- **🎥 Generate meeting notes** from Google Meet recordings
- **📜 Analyze contracts** for risks and compliance issues

### **Next Steps:**

1. **📚 Read the [Technical Documentation](README_detail.md)** for advanced features
2. **🔧 Customize agent settings** for your specific needs
3. **📊 Monitor performance metrics** to optimize automation
4. **🤝 Join our community** for support and feature updates

### **Need Help?**

- 📖 **Documentation**: [README_detail.md](README_detail.md)
- 🐛 **Issues**: GitHub Issues page
- 💬 **Community**: Discord server
- 📧 **Enterprise Support**: Contact our team

---

**Happy Automating! 🚀**
