# 📧 AI-Powered Email Automation Agent

A **sophisticated, intelligent email management system** that combines AI-powered classification, automated meeting scheduling, and smart response generation to streamline your email workflow. This agent integrates Google Gemini AI with Gmail and Google Calendar APIs to provide enterprise-grade email automation with human oversight.

## 🚀 **Advanced AI Features**

### 🧠 **AI-Powered Email Intelligence**
- **🎯 Smart Email Classification**: Gemini-powered categorization into 7 distinct types
- **📊 Confidence-Based Processing**: Only acts on high-confidence classifications (configurable threshold)
- **🔍 Context-Aware Analysis**: Analyzes subject, sender, and body content for accurate classification
- **📈 Adaptive Learning**: Tracks performance metrics for continuous improvement
- **🎨 Template Generation**: AI-powered personalization of response templates

### 🗓️ **Intelligent Meeting Scheduling**
- **⚡ Real-Time Calendar Integration**: Live Google Calendar availability checking
- **🕒 Smart Time Slot Detection**: Finds optimal meeting times based on working hours (9 AM - 6 PM IST)
- **🌍 Time Zone Intelligence**: Built-in IST (Asia/Kolkata) timezone support
- **📅 Automated Event Creation**: Google Meet links and calendar invites generated automatically
- **⏰ Conflict Avoidance**: Intelligent overlap detection with existing calendar events
- **📋 Meeting Details Extraction**: AI-powered parsing of duration, preferences, and meeting type

### 🤖 **Enterprise-Grade Automation**
- **🔄 Multi-Service Integration**: Seamless Gmail + Calendar + Gemini AI workflow
- **🛡️ OAuth2 Security**: Secure Google API authentication with token management
- **📊 Performance Analytics**: Comprehensive metrics tracking and reporting
- **⚖️ Human-in-the-Loop**: Interactive approval workflow for all automated actions
- **🎯 Selective Processing**: Configurable lookback periods and confidence thresholds
- **📝 Action Logging**: Complete audit trail of all agent decisions and user interactions

## 📋 **Email Categories & Capabilities**

### **🎯 Supported Email Types**
1. **📅 MEETING_REQUEST**: New meeting, call, or appointment requests
2. **💰 INVOICE**: Bills, receipts, payment requests with template responses
3. **🔄 SCHEDULING**: Meeting replies, calendar updates, rescheduling
4. **❓ FAQ**: Common questions with intelligent template-based answers
5. **📰 NEWSLETTER**: Automated promotional/informational emails (filtered)
6. **🚫 SPAM**: Unsolicited commercial emails (flagged and skipped)
7. **📩 OTHER**: Miscellaneous emails requiring manual review

### **⚡ Automated Actions**
- **📅 Meeting Scheduling**: Finds 3 best available slots, creates calendar events with Meet links
- **📝 Template Responses**: AI-generated, personalized replies for invoices, FAQs, scheduling
- **🔄 Thread Management**: Maintains email conversation context and reply threading
- **📊 Performance Tracking**: Logs user acceptance rates, automation success, and edit frequency

## 🚀 **Quick Start**

### **🌐 Web Interface (Recommended)**

The easiest way to use the Email Agent is through the AgenticSuite web interface:

```bash
# Start the AgenticSuite platform
cd backend
python app.py

# Open browser to http://localhost:5000
# Click "Email Agent" to access the web interface
```

**Web Interface Features:**
- 📊 **Real-time Processing**: Watch emails being classified and processed live
- 🎯 **Interactive Approval**: Review and approve AI-generated responses with one click
- 📈 **Live Metrics**: View automation performance and success rates
- 🔄 **Batch Processing**: Handle multiple emails efficiently
- 📱 **Mobile Responsive**: Manage emails from any device

### **💻 Command Line Usage**

For advanced users and automation scenarios:

```python
from email_agent import EmailAgent

# Initialize with custom settings
agent = EmailAgent(
    lookback_hours=72,      # Process emails from last 3 days
    confidence_threshold=0.7 # Only act on 70%+ confidence classifications
)

# Process emails interactively
agent.process_emails()

# Check performance metrics
metrics = agent.get_metrics()
print(f"Automation rate: {metrics['automation_rate']:.1%}")
```

### **⚙️ Setup Requirements**

**Complete setup instructions available in [AgenticSuite Setup Guide](../SETUP.md)**

Quick setup checklist:
- ✅ Python 3.8+ installed
- ✅ Google Cloud Platform project created
- ✅ Gmail and Calendar APIs enabled
- ✅ OAuth2 credentials configured
- ✅ Gemini API key obtained
- ✅ Dependencies installed via `pip install -r requirements.txt`

## 🔬 **Technical Deep Dive**

### **🎯 AI Classification Pipeline**

```python
# Email classification with confidence scoring
classification_prompt = f"""
Analyze email and classify into categories:
- MEETING_REQUEST, INVOICE, SCHEDULING, FAQ, NEWSLETTER, SPAM, OTHER

Subject: {email.subject}
From: {email.sender}  
Body: {email.body[:1000]}

Respond: CATEGORY,confidence_score
Example: MEETING_REQUEST,0.95
"""

# Gemini processes and returns category + confidence
result = gemini_model.generate_content(classification_prompt)
category, confidence = parse_response(result.text)
```

### **🗓️ Smart Calendar Integration**

```python
# Intelligent time slot finding algorithm
def find_available_slots(duration_minutes, days_ahead):
    # 1. Get busy times from Google Calendar API
    busy_periods = get_calendar_busy_times(days_ahead)
    
    # 2. Generate working hour slots (9 AM - 6 PM IST)
    working_slots = generate_working_slots(days_ahead)
    
    # 3. Filter out conflicts and past times
    available_slots = filter_conflicts(working_slots, busy_periods)
    
    # 4. Return top options optimized for user preference
    return available_slots[:10]
```

### **📝 AI-Powered Response Generation**

```python
# Template-based response with AI personalization
response_prompt = f"""
Fill email template based on original email context:

Original Email: {email.content}
Template: {category_template}

Provide natural, personalized response maintaining professional tone.
"""

# Gemini generates contextually appropriate response
response = gemini_model.generate_content(response_prompt)
```

## 📊 **Performance Analytics & Metrics**

### **📈 Tracked Metrics**
- **📧 Total Emails Processed**: Volume of emails analyzed
- **🤖 Automation Rate**: Percentage of emails handled automatically
- **📅 Meetings Scheduled**: Successfully created calendar events
- **✅ Draft Acceptance Rate**: User approval rate for AI-generated responses
- **✏️ Edit Frequency**: How often users modify AI drafts
- **🎯 Classification Confidence**: Average confidence scores by category

### **📋 Performance Dashboard**
```python
# Get comprehensive analytics
metrics = agent.get_metrics()

print(f"""
📊 Email Agent Performance:
   📧 Emails Processed: {metrics['emails_processed']}
   🤖 Automation Rate: {metrics['automation_rate']:.1%}
   📅 Meetings Scheduled: {metrics['meetings_scheduled']}
   ✅ Drafts Accepted: {metrics['drafts_accepted']}
""")
```

## 🎯 **Advanced Use Cases**

### **🏢 Enterprise Workflow Integration**

```python
# Custom business hours and preferences
agent = EmailAgent(
    lookback_hours=168,        # Process weekly backlog
    confidence_threshold=0.8,  # Higher confidence for enterprise use
)

# Batch processing for high-volume scenarios
for time_period in business_hours:
    emails = agent.fetch_recent_emails()
    processed = agent.process_emails_batch(emails)
    generate_daily_report(processed)
```

### **⚙️ Custom Classification Rules**

```python
# Extend categories for industry-specific needs
class CustomEmailAgent(EmailAgent):
    CUSTOM_CATEGORIES = {
        "SUPPORT_TICKET": "customer support requests",
        "SALES_INQUIRY": "potential sales opportunities", 
        "COMPLIANCE": "regulatory and compliance emails"
    }
    
    def classify_email(self, email):
        # Add custom classification logic
        return enhanced_classification(email)
```

### **🔄 Automated Workflow Chains**

```python
# Multi-step automation workflows
def automated_meeting_flow(email):
    # 1. Classify as meeting request
    classification = agent.classify_email(email)
    
    # 2. Extract meeting requirements
    meeting_details = agent.parse_meeting_details(email)
    
    # 3. Find optimal time slots
    slots = agent.find_available_slots(meeting_details['duration'])
    
    # 4. Create calendar event with best slot
    event_id = agent.create_calendar_event(email, slots[0])
    
    # 5. Send confirmation with Meet link
    confirmation = agent.generate_meeting_confirmation(email, slots[0])
    agent.send_email(email.sender, "Meeting Confirmed", confirmation)
    
    # 6. Log success metrics
    agent.log_action(email.id, "MEETING_REQUEST", "automated_success")
```

## 🛠️ **Configuration & Customization**

### **⚙️ Agent Configuration**

```python
# Flexible initialization options
agent = EmailAgent(
    lookback_hours=72,           # Email processing window
    confidence_threshold=0.7,    # Minimum confidence for action
    working_hours=(9, 18),       # Business hours in IST
    max_slots_to_find=10,        # Calendar availability options
    timezone='Asia/Kolkata'      # Default timezone
)
```

### **📝 Custom Templates**

```python
# Add custom response templates
agent.TEMPLATES[EmailCategory.SUPPORT] = """
Hi {sender_name},

Thank you for contacting our support team. 
Your ticket #{ticket_id} has been created.

Our team will respond within 24 hours.

Best regards,
Support Team
"""
```

### **🔄 Integration Hooks**

```python
# Custom processing hooks
class EnterpriseEmailAgent(EmailAgent):
    def post_classification_hook(self, email, classification):
        # Log to enterprise monitoring system
        self.enterprise_logger.log_classification(email, classification)
    
    def pre_send_hook(self, email_draft):
        # Apply enterprise policies
        return self.compliance_checker.validate(email_draft)
```

## 📋 **Human-in-the-Loop Workflow**

### **🎯 Interactive Processing**
The agent implements a sophisticated approval workflow:

1. **📧 Email Fetching**: Retrieves emails from configurable lookback period
2. **🎯 AI Classification**: Gemini analyzes and categorizes with confidence score
3. **⚖️ Confidence Filtering**: Only processes high-confidence classifications
4. **🤖 Action Generation**: Creates appropriate response or calendar event
5. **👤 Human Review**: Presents draft for user approval/editing
6. **✅ User Decision**: Send as-is, edit, or cancel
7. **📊 Metrics Logging**: Records user decisions for system improvement

### **🔄 User Interaction Flow**

```
📧 New Email Detected
    ↓
🎯 AI Classification (with confidence)
    ↓
⚖️ Confidence > Threshold?
    ↓ YES
🤖 Generate Response/Action
    ↓
👤 Present to User:
   • [S]end
   • [E]dit  
   • [C]ancel
    ↓
📊 Log Decision & Metrics
```

## 🔧 **System Architecture**

### **🏗️ Core Components**

```
EmailAgent (Main Orchestrator)
├── 🔐 GoogleAuthenticator (OAuth2 + Token Management)
├── 📧 GmailService (Email Fetching + Sending)
├── 📅 CalendarService (Event Creation + Availability)
├── 🧠 GeminiClassifier (AI Email Analysis)
├── 📝 TemplateGenerator (Response Creation)
├── 📊 MetricsTracker (Performance Analytics)
└── ⚙️ InteractiveProcessor (Human-in-Loop Workflow)
```

### **🔄 Data Flow Architecture**

```
📧 Gmail API (Email Retrieval)
    ↓
🎯 Gemini AI (Classification + Content Analysis)
    ↓
📅 Calendar API (Availability Check + Event Creation)
    ↓
📝 Response Generation (Template + AI Personalization)
    ↓
👤 Human Approval (Interactive Review)
    ↓
📧 Gmail API (Sending Responses)
    ↓
📊 CSV Metrics (Performance Tracking)
```

## 🚀 **Enterprise Features**

### **🛡️ Security & Compliance**
- **🔐 OAuth2 Authentication**: Secure Google API access with token refresh
- **🔒 Credential Management**: Encrypted token storage and rotation
- **📋 Audit Trail**: Complete logging of all actions and decisions
- **⚖️ Privacy-First**: No email content stored permanently
- **🛡️ API Rate Limiting**: Intelligent request throttling

### **📊 Analytics & Monitoring**
- **📈 Real-Time Metrics**: Live tracking of automation performance
- **📋 CSV Export**: Detailed analytics for enterprise reporting
- **🎯 Success Rates**: Classification accuracy and user acceptance metrics
- **⏱️ Processing Time**: Performance benchmarking and optimization
- **🔄 User Behavior**: Edit patterns and preference learning

### **⚡ Scalability Features**
- **🔄 Batch Processing**: Handle high-volume email scenarios
- **⏰ Configurable Scheduling**: Flexible processing windows
- **🎯 Smart Filtering**: Focus on actionable emails only
- **💾 Efficient Storage**: Minimal local data footprint
- **🚀 API Optimization**: Intelligent request batching and caching

## 🔮 **Future Enhancements**

### **🚀 Advanced AI Integration**
- **📚 RAG Implementation**: Retrieve similar email responses for better context
- **🧠 Multi-Model Ensemble**: Combine different AI models for better accuracy
- **🎯 Few-Shot Learning**: Rapid adaptation to user-specific email patterns
- **📊 Sentiment Analysis**: Emotion-aware response generation
- **🔍 Intent Recognition**: Advanced email purpose detection

### **🏢 Enterprise Extensions**
- **🌐 Multi-Language Support**: International email processing
- **🔗 CRM Integration**: Salesforce, HubSpot, and custom system connections
- **📱 Mobile Interface**: Native iOS/Android apps for on-the-go management
- **🤖 Slack/Teams Bots**: Integrated workflow notifications
- **📊 Advanced Analytics**: Machine learning-powered insights and predictions

### **🎨 User Experience**
- **💻 Web Dashboard**: Browser-based management interface
- **📱 Progressive Web App**: Mobile-optimized user experience
- **🎨 Customizable UI**: Themes and layout personalization
- **🔔 Smart Notifications**: Intelligent priority-based alerts
- **📈 Performance Insights**: Personalized productivity analytics

## 🏆 **Technical Achievements**

**🎯 This email agent represents a production-ready implementation of advanced automation techniques:**

- ✅ **AI-Powered Classification** with confidence-based processing
- ✅ **Real-Time Calendar Integration** with conflict detection
- ✅ **Human-in-the-Loop Automation** with approval workflows
- ✅ **Multi-Service API Integration** (Gmail + Calendar + Gemini)
- ✅ **Enterprise Security** with OAuth2 and audit trails
- ✅ **Performance Analytics** with comprehensive metrics tracking
- ✅ **Intelligent Time Management** with timezone support
- ✅ **Template-Based Responses** with AI personalization

## 🛠️ **Troubleshooting**

### **🔧 Common Issues**

**🔐 Authentication Problems**
```bash
# Clear existing tokens
rm token.json

# Re-run authentication
python setup_auth.py

# Verify API permissions in Google Cloud Console
```

**📧 Email Fetching Issues**
```bash
# Check Gmail API quota
# Verify OAuth2 scopes include gmail.readonly
# Test with smaller lookback window
agent = EmailAgent(lookback_hours=24)
```

**📅 Calendar Integration Problems**
```bash
# Verify Calendar API is enabled
# Check calendar permissions
# Test timezone settings
import pytz
print(pytz.timezone('Asia/Kolkata'))
```

### **📊 Performance Optimization**

**🚀 High-Volume Processing**
```python
# Optimize for large email volumes
agent = EmailAgent(
    lookback_hours=168,      # Weekly processing
    confidence_threshold=0.8, # Higher precision
    max_results=100          # Batch processing
)
```

**💰 API Cost Management**
- Monitor Gemini API usage in Google Cloud Console
- Use confidence thresholds to reduce unnecessary AI calls
- Implement local caching for repeated classifications
- Batch API requests where possible

## 📄 **License & Disclaimer**

This tool is for personal and educational use. Ensure compliance with your organization's email policies and Google's API terms of service.

---

## 🏆 **Key Capabilities Summary**

**🚀 Built with**: Google Gemini AI, Gmail API, Google Calendar API, OAuth2 Security, Python asyncio, and enterprise-grade error handling for reliable email automation.

**🎯 Perfect for**: Busy professionals, small business owners, customer support teams, and anyone looking to automate routine email tasks while maintaining human oversight and control.

**🔬 Technical Foundation**: Implements modern AI techniques including confidence-based classification, template-driven responses, intelligent scheduling algorithms, and comprehensive performance analytics.

---

## 🔗 **Related Documentation**

- **[AgenticSuite Main Documentation](../README.md)** - Project overview and introduction
- **[Complete Setup Guide](../SETUP.md)** - Detailed installation and configuration instructions  
- **[Technical Documentation](../README_detail.md)** - Architecture and development guide
- **[Meeting Agent](MEETING_AGENT_README.md)** - Automated meeting notes generation
- **[Contract Agent](CONTRACT_AGENT_README.md)** - AI-powered contract risk analysis

**💬 Support & Community**
- 🐛 Report issues on GitHub
- 💡 Request features via GitHub Issues
- 📧 Enterprise support available

*Part of the AgenticSuite AI Automation Platform*