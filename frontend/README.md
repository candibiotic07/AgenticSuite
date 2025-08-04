# AgenticSuite Frontend

A modern web interface for the AgenticSuite AI automation platform.

## Features

- **Dashboard** - Overview of all AI agents with quick stats
- **Email Agent Interface** - Configure and run email automation
- **Meeting Agent Interface** - Process Google Meet recordings into AI notes
- **Contract Agent Interface** - Upload and analyze contracts for risks
- **Authentication Flow** - Secure Google OAuth integration
- **Real-time Updates** - Live status updates and progress tracking

## Quick Start

1. **Install Dependencies** (if not already done):
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set up Authentication**:
   ```bash
   cd backend
   python auth.py
   ```

3. **Start the Web Server**:
   ```bash
   cd backend
   python app.py
   ```

4. **Access the Web Interface**:
   Open your browser to: `http://localhost:5000`

## File Structure

```
frontend/
├── templates/          # HTML templates
│   ├── base.html      # Base template with navigation
│   ├── dashboard.html # Main dashboard
│   ├── auth.html      # Authentication page
│   ├── email.html     # Email agent interface
│   ├── meeting.html   # Meeting agent interface
│   └── contract.html  # Contract agent interface
├── static/
│   ├── css/
│   │   └── style.css  # Complete CSS styling
│   └── js/
│       └── app.js     # JavaScript functionality
└── README.md          # This file
```

## Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Icons**: Font Awesome 6
- **Styling**: Custom CSS with CSS Variables
- **API**: RESTful API with JSON responses

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development

The frontend is designed to work seamlessly with the existing backend agents without modifying their core functionality. All agent logic remains unchanged - the web interface simply provides a modern UI wrapper around the existing command-line tools.

### Key Features:

1. **Responsive Design** - Works on desktop, tablet, and mobile
2. **Dark/Light Theme Support** - Ready for theme switching
3. **Real-time Updates** - Progress tracking and live notifications
4. **File Upload** - Drag & drop contract upload with progress
5. **Keyboard Shortcuts** - Quick navigation (press `?` for help)
6. **Error Handling** - Comprehensive error handling and user feedback

### API Endpoints:

- `GET /` - Dashboard
- `GET /email` - Email agent interface  
- `GET /meeting` - Meeting agent interface
- `GET /contract` - Contract agent interface
- `POST /api/authenticate` - Start OAuth flow
- `POST /api/email/process` - Process emails
- `POST /api/meeting/process` - Process meeting
- `POST /api/contract/upload` - Upload & analyze contract
- `GET /api/*/metrics` - Get agent metrics

## Security

- All authentication is handled through Google OAuth
- Credentials stored locally only
- No data sent to external servers
- User approval required for all AI actions