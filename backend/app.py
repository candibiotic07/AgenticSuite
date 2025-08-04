"""
AgenticSuite Web API
Flask application that wraps the existing agents for web interface
"""

from flask import Flask, render_template, request, jsonify, session, send_from_directory, redirect, url_for
from flask_cors import CORS
import os
import json
import logging
import threading
import time
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
import uuid

# Import existing agents (without modifying them)
try:
    from email_agent import EmailAgent
    from meeting_agent import MeetingNotesAutomation
    from contract_agent import ContractAgent
    from auth import authenticate_google
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    EmailAgent = None
    MeetingNotesAutomation = None
    ContractAgent = None

app = Flask(__name__, 
           template_folder='../frontend/templates',
           static_folder='../frontend/static')
app.secret_key = os.urandom(24)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for agent instances
email_agent = None
meeting_agent = None
contract_agent = None

# Global storage for pending email drafts and processing status
pending_drafts = {}
processing_status = {
    'active': False,
    'current_email': None,
    'progress': {'current': 0, 'total': 0},
    'logs': []
}

# Global state for meeting processing
meeting_processing_status = {
    'active': False,
    'current_step': '',
    'progress': {'current': 0, 'total': 6},  # 6 main steps in meeting processing
    'logs': [],
    'result': None
}

# Global state for contract processing
contract_processing_status = {
    'active': False,
    'current_step': '',
    'progress': {'current': 0, 'total': 6},  # 6 main steps: upload, extract, segment, analyze, risk, report
    'logs': [],
    'result': None
}

# Upload folder for contracts
UPLOAD_FOLDER = 'contractDATAtemp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for contracts
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_auth():
    """Check if authentication is valid"""
    try:
        if os.path.exists('token.json'):
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            
            # Load credentials
            creds = Credentials.from_authorized_user_file('token.json')
            
            if creds:
                # If expired but has refresh token, try to refresh
                if creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        # Save refreshed credentials
                        with open('token.json', 'w') as token:
                            token.write(creds.to_json())
                        logger.info("Credentials refreshed successfully")
                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh credentials: {refresh_error}")
                        return False
                
                # Check if credentials are valid
                if creds.valid:
                    return True
    except Exception as e:
        logger.error(f"Auth check failed: {e}")
    return False

def initialize_agents():
    """Initialize all agents if authenticated"""
    global email_agent, meeting_agent, contract_agent
    
    if not check_auth():
        logger.error("Cannot initialize agents: Authentication failed")
        return False
    
    try:
        # Initialize Email Agent
        if EmailAgent and not email_agent:
            try:
                email_agent = EmailAgent(lookback_hours=72)
                logger.info("Email agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Email Agent: {e}")
        
        # Initialize Meeting Agent
        if MeetingNotesAutomation and not meeting_agent:
            try:
                meeting_agent = MeetingNotesAutomation()
                logger.info("Meeting agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Meeting Agent: {e}")
        
        # Initialize Contract Agent
        if ContractAgent and not contract_agent:
            try:
                contract_agent = ContractAgent()
                logger.info("Contract agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Contract Agent: {e}")
        
        logger.info("Agent initialization completed")
        return True
    except Exception as e:
        logger.error(f"Critical error during agent initialization: {e}")
        return False

def process_emails_web_mode():
    """Web-compatible email processing that doesn't use CLI input"""
    global email_agent, processing_status, pending_drafts
    
    try:
        # Add log
        processing_status['logs'].append({
            'type': 'info',
            'message': 'Starting email processing...',
            'timestamp': datetime.now().isoformat()
        })
        
        # Fetch emails
        emails = email_agent.fetch_recent_emails()
        
        if not emails:
            processing_status.update({
                'active': False,
                'logs': processing_status['logs'] + [{
                    'type': 'info',
                    'message': 'No new emails to process',
                    'timestamp': datetime.now().isoformat()
                }]
            })
            return
        
        processing_status['progress']['total'] = len(emails)
        processing_status['logs'].append({
            'type': 'info',
            'message': f'Found {len(emails)} emails to process',
            'timestamp': datetime.now().isoformat()
        })
        
        for i, email in enumerate(emails, 1):
            processing_status['progress']['current'] = i
            processing_status['current_email'] = {
                'sender': email.sender,
                'subject': email.subject
            }
            
            processing_status['logs'].append({
                'type': 'info',
                'message': f'[{i}/{len(emails)}] Processing email from: {email.sender}',
                'timestamp': datetime.now().isoformat()
            })
            
            # Classify email
            classification = email_agent.classify_email(email)
            
            if classification.confidence < email_agent.confidence_threshold:
                processing_status['logs'].append({
                    'type': 'warning',
                    'message': f'Low confidence classification ({classification.confidence:.2f}). Skipping...',
                    'timestamp': datetime.now().isoformat()
                })
                email_agent.log_action(email.message_id, classification.category.value, 
                                     "skipped", False, False, False)
                continue
            
            processing_status['logs'].append({
                'type': 'info',
                'message': f'Classified as: {classification.category.value} (confidence: {classification.confidence:.2f})',
                'timestamp': datetime.now().isoformat()
            })
            
            # Handle based on category
            if classification.category.value == 'MEETING_REQUEST':
                handle_meeting_request_web(email)
            elif classification.category.value in ['INVOICE', 'FAQ', 'SCHEDULING']:
                handle_template_response_web(email, classification.category)
            elif classification.category.value in ['NEWSLETTER', 'SPAM']:
                # Archive newsletters and spam emails
                archived = email_agent.archive_email(email.message_id)
                if archived:
                    processing_status['logs'].append({
                        'type': 'info',
                        'message': f'{classification.category.value} email archived automatically',
                        'timestamp': datetime.now().isoformat()
                    })
                    email_agent.log_action(email.message_id, classification.category.value, 
                                         "archived", False, False, False)
                else:
                    processing_status['logs'].append({
                        'type': 'warning',
                        'message': f'Failed to archive {classification.category.value} email',
                        'timestamp': datetime.now().isoformat()
                    })
                    email_agent.log_action(email.message_id, classification.category.value, 
                                         "archive_failed", False, False, False)
            else:
                processing_status['logs'].append({
                    'type': 'info',
                    'message': f'Category {classification.category.value} - No action needed',
                    'timestamp': datetime.now().isoformat()
                })
                email_agent.log_action(email.message_id, classification.category.value, 
                                     "no_action", False, False, False)
        
        # Final status
        processing_status.update({
            'active': False,
            'current_email': None,
            'logs': processing_status['logs'] + [{
                'type': 'success',
                'message': 'Email processing completed!',
                'timestamp': datetime.now().isoformat()
            }]
        })
        
    except Exception as e:
        processing_status.update({
            'active': False,
            'current_email': None,
            'logs': processing_status['logs'] + [{
                'type': 'error',
                'message': f'Processing failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }]
        })

def handle_template_response_web(email, category):
    """Handle template responses in web mode"""
    global pending_drafts, processing_status
    
    try:
        # Generate response
        response = email_agent.generate_template_response(email, category)
        
        if not response:
            processing_status['logs'].append({
                'type': 'error',
                'message': 'Failed to generate response',
                'timestamp': datetime.now().isoformat()
            })
            email_agent.log_action(email.message_id, category.value, "generation_failed", 
                                 False, False, False)
            return
        
        # Store draft for approval
        draft_id = str(uuid.uuid4())
        pending_drafts[draft_id] = {
            'email': email,
            'category': category.value,
            'draft': response,
            'created': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        processing_status['logs'].append({
            'type': 'draft',
            'message': f'Draft generated for {category.value} response - awaiting approval',
            'timestamp': datetime.now().isoformat(),
            'draft_id': draft_id,
            'draft_preview': response[:100] + '...' if len(response) > 100 else response
        })
        
    except Exception as e:
        processing_status['logs'].append({
            'type': 'error',
            'message': f'Error generating template response: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

def handle_meeting_request_web(email):
    """Handle meeting requests in web mode"""
    global pending_drafts, processing_status
    
    try:
        # Get meeting details and available time slots (same as CLI version)
        available_slots = email_agent.handle_meeting_request(email)
        
        if not available_slots:
            processing_status['logs'].append({
                'type': 'warning',
                'message': 'No available time slots found',
                'timestamp': datetime.now().isoformat()
            })
            email_agent.log_action(email.message_id, 'MEETING_REQUEST', 
                                 "no_slots", False, False, False)
            return
        
        # Store draft for approval with all available slots
        draft_id = str(uuid.uuid4())
        pending_drafts[draft_id] = {
            'email': email,
            'category': 'MEETING_REQUEST',
            'available_slots': [
                {
                    'start': slot.start.isoformat(),
                    'end': slot.end.isoformat(),
                    'display': slot.start.strftime('%a, %b %d at %I:%M %p') + ' IST'
                } for slot in available_slots
            ],
            'selected_slot_index': None,  # To be set when user selects
            'created': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        slots_display = ', '.join([slot.start.strftime('%a, %b %d at %I:%M %p') for slot in available_slots[:3]])
        
        processing_status['logs'].append({
            'type': 'draft',
            'message': f'Meeting request detected - {len(available_slots)} time slots available for selection',
            'timestamp': datetime.now().isoformat(),
            'draft_id': draft_id,
            'draft_preview': f'Available slots: {slots_display}'
        })
        
    except Exception as e:
        processing_status['logs'].append({
            'type': 'error',
            'message': f'Error handling meeting request: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })
        email_agent.log_action(email.message_id, 'MEETING_REQUEST', 
                             "error", False, False, False)

# Main Routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    auth_status = check_auth()
    if not auth_status:
        # If not authenticated, redirect to auth page
        return redirect(url_for('auth_page'))
    
    # Initialize agents if authenticated
    if not email_agent or not meeting_agent or not contract_agent:
        initialize_agents()
    
    return render_template('dashboard.html', 
                         auth_status=auth_status,
                         email_available=email_agent is not None,
                         meeting_available=meeting_agent is not None,
                         contract_available=contract_agent is not None)

@app.route('/auth')
def auth_page():
    """Authentication page"""
    # If already authenticated, redirect to dashboard
    if check_auth():
        return redirect(url_for('dashboard'))
    return render_template('auth.html')

# API Routes
@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    """Handle authentication"""
    try:
        if authenticate_google:
            creds = authenticate_google()
            if creds:
                if initialize_agents():
                    return jsonify({'success': True, 'message': 'Authentication successful'})
                else:
                    return jsonify({'success': False, 'message': 'Authentication successful but agent initialization failed'})
            else:
                return jsonify({'success': False, 'message': 'Authentication failed'})
        else:
            return jsonify({'success': False, 'message': 'Authentication module not available'})
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/auth/status')
def auth_status_api():
    """Get authentication status"""
    return jsonify({'authenticated': check_auth()})

# Email Agent Routes
@app.route('/email')
def email_dashboard():
    """Email agent dashboard"""
    auth_status = check_auth()
    if not auth_status:
        return redirect(url_for('auth_page'))
    
    # Initialize agents if not already done
    if not email_agent:
        initialize_agents()
    
    return render_template('email.html', auth_status=auth_status, email_available=email_agent is not None)

@app.route('/api/email/process', methods=['POST'])
def process_emails():
    """Process emails with the email agent (web-compatible version)"""
    global email_agent, processing_status, pending_drafts
    
    if not email_agent and EmailAgent:
        initialize_agents()
    
    if not email_agent:
        return jsonify({'success': False, 'message': 'Email agent not available'})
    
    if processing_status['active']:
        return jsonify({'success': False, 'message': 'Email processing already in progress'})
    
    try:
        data = request.get_json() or {}
        lookback_hours = data.get('lookback_hours', 72)
        confidence_threshold = data.get('confidence_threshold', 0.7)
        
        # Update agent configuration
        email_agent.lookback_hours = lookback_hours
        email_agent.confidence_threshold = confidence_threshold
        
        # Clear previous state
        pending_drafts.clear()
        processing_status.update({
            'active': True,
            'current_email': None,
            'progress': {'current': 0, 'total': 0},
            'logs': []
        })
        
        # Process emails in web-compatible mode
        def process_emails_web_async():
            try:
                process_emails_web_mode()
            except Exception as e:
                logger.error(f"Email processing error: {e}")
                processing_status.update({
                    'active': False,
                    'current_email': None,
                    'logs': processing_status['logs'] + [{'type': 'error', 'message': f'Processing failed: {str(e)}', 'timestamp': datetime.now().isoformat()}]
                })
        
        thread = threading.Thread(target=process_emails_web_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Email processing started'})
        
    except Exception as e:
        logger.error(f"Email processing error: {e}")
        processing_status['active'] = False
        return jsonify({'success': False, 'message': str(e)})

def calculate_email_metrics():
    """Calculate comprehensive email metrics from CSV data"""
    import csv
    import os
    from collections import defaultdict
    
    metrics_file = 'mailDATAtemp/data/email_agent_metrics.csv'
    
    # Default metrics
    metrics = {
        'total_emails': 0,
        'meetings_scheduled': 0,
        'responses_generated': 0,
        'acceptance_rate': 0.0,
        'estimated_time_saved': 0.0
    }
    
    if not os.path.exists(metrics_file):
        return metrics
    
    try:
        with open(metrics_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return metrics
        
        total_drafts_generated = 0
        drafts_accepted_without_edit = 0
        meetings_scheduled = 0
        
        # Time saved estimates (in minutes)
        time_saved_per_action = {
            'FAQ': 5,           # 5 minutes to research and respond to FAQ
            'INVOICE': 3,       # 3 minutes to acknowledge invoice
            'MEETING_REQUEST': 5,  # 5 minutes to check calendar and respond
            'SCHEDULING': 4,    # 4 minutes to coordinate scheduling
            'NEWSLETTER': 1,    # 1 minute to read, classify and archive
            'SPAM': 1,          # 1 minute to read, classify and archive  
            'OTHER': 1,         # 1 minute to read, classify and archive
        }
        
        estimated_time_saved = 0.0
        
        for row in rows:
            email_id = row['email_id']
            category = row['category']
            action_taken = row['action_taken']
            draft_accepted = row['draft_accepted'].lower() == 'true'
            meeting_scheduled_flag = row['meeting_scheduled'].lower() == 'true'
            user_edited = row['user_edited'].lower() == 'true'
            
            # Count meetings scheduled
            if meeting_scheduled_flag:
                meetings_scheduled += 1
            
            # Count drafts generated (sent or cancelled means a draft was created)
            if action_taken in ['sent', 'cancelled']:
                total_drafts_generated += 1
                
                # Count drafts accepted without user edit
                if draft_accepted and not user_edited:
                    drafts_accepted_without_edit += 1
            
            # Calculate time saved for ALL processed emails (including classification-only)
            # Time is saved whenever the agent processes an email, regardless of action
            if action_taken in ['sent', 'cancelled', 'no_action', 'skipped', 'archived', 'archive_failed']:
                time_saved = time_saved_per_action.get(category, 1)  # Default 1 minute
                estimated_time_saved += time_saved
        
        # Calculate final metrics - count total email parsing actions (lines in CSV)
        metrics['total_emails'] = len(rows)  # Each row = one email parsed by agent
        metrics['meetings_scheduled'] = meetings_scheduled
        metrics['responses_generated'] = total_drafts_generated
        
        # Calculate acceptance rate
        if total_drafts_generated > 0:
            metrics['acceptance_rate'] = round((drafts_accepted_without_edit / total_drafts_generated) * 100, 1)
        
        # Return time saved in minutes
        metrics['estimated_time_saved'] = round(estimated_time_saved, 0)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics from CSV: {e}")
        return metrics

@app.route('/api/email/metrics')
def email_metrics():
    """Get comprehensive email agent metrics"""
    try:
        metrics = calculate_email_metrics()
        return jsonify({'success': True, 'metrics': metrics})
    except Exception as e:
        logger.error(f"Failed to get email metrics: {e}")
        return jsonify({'success': True, 'metrics': {
            'total_emails': 0,
            'meetings_scheduled': 0,
            'responses_generated': 0,
            'acceptance_rate': 0.0,
            'estimated_time_saved': 0.0
        }})

@app.route('/api/email/status')
def email_processing_status():
    """Get current email processing status and logs"""
    global processing_status
    return jsonify({
        'success': True,
        'status': processing_status
    })

@app.route('/api/email/drafts')
def get_pending_drafts():
    """Get all pending email drafts"""
    global pending_drafts
    
    # Convert drafts to a format suitable for frontend
    drafts_list = []
    for draft_id, draft_data in pending_drafts.items():
        if draft_data['status'] == 'pending':
            # Convert email object to dict for JSON serialization
            email_dict = {
                'sender': draft_data['email'].sender,
                'subject': draft_data['email'].subject,
                'message_id': draft_data['email'].message_id
            }
            
            draft_item = {
                'id': draft_id,
                'email': email_dict,
                'category': draft_data['category'],
                'created': draft_data['created']
            }
            
            # Handle different draft types
            if draft_data['category'] == 'MEETING_REQUEST':
                draft_item['available_slots'] = draft_data.get('available_slots', [])
                draft_item['selected_slot_index'] = draft_data.get('selected_slot_index')
            else:
                draft_item['draft'] = draft_data.get('draft', '')
            
            drafts_list.append(draft_item)
    
    return jsonify({
        'success': True,
        'drafts': drafts_list
    })

@app.route('/api/email/drafts/<draft_id>/approve', methods=['POST'])
def approve_draft(draft_id):
    """Approve and send a draft email or schedule meeting"""
    global pending_drafts, email_agent
    
    if draft_id not in pending_drafts:
        return jsonify({'success': False, 'message': 'Draft not found'})
    
    draft_data = pending_drafts[draft_id]
    if draft_data['status'] != 'pending':
        return jsonify({'success': False, 'message': 'Draft already processed'})
    
    try:
        data = request.get_json() or {}
        email = draft_data['email']
        sender_email = email.sender.split('<')[-1].strip('>')
        
        if draft_data['category'] == 'MEETING_REQUEST':
            # Handle meeting request approval
            selected_slot_index = data.get('selected_slot_index')
            
            if selected_slot_index is None:
                return jsonify({'success': False, 'message': 'Please select a time slot'})
            
            available_slots = draft_data.get('available_slots', [])
            if selected_slot_index < 0 or selected_slot_index >= len(available_slots):
                return jsonify({'success': False, 'message': 'Invalid time slot selection'})
            
            selected_slot_data = available_slots[selected_slot_index]
            
            # Create TimeSlot object for calendar event creation
            from datetime import datetime
            import pytz
            ist = pytz.timezone('Asia/Kolkata')
            
            # Parse the ISO datetime strings
            start_dt = datetime.fromisoformat(selected_slot_data['start'])
            end_dt = datetime.fromisoformat(selected_slot_data['end'])
            
            # Create a TimeSlot-like object
            class TimeSlotObj:
                def __init__(self, start, end):
                    self.start = start
                    self.end = end
            
            selected_slot = TimeSlotObj(start_dt, end_dt)
            
            # Parse meeting details
            meeting_details = email_agent._parse_meeting_details(email)
            
            # Create calendar event
            print(f"\n📅 Creating calendar event for {selected_slot_data['display']}...")
            event_id = email_agent.create_calendar_event(email, selected_slot, meeting_details)
            
            if not event_id:
                return jsonify({'success': False, 'message': 'Failed to create calendar event'})
            
            # Generate and send confirmation email
            confirmation = email_agent.generate_meeting_confirmation(email, selected_slot)
            
            success = email_agent.send_email(
                sender_email, 
                f"Re: {email.subject}", 
                confirmation, 
                email.thread_id
            )
            
            if success:
                # Update draft status
                pending_drafts[draft_id]['status'] = 'sent'
                pending_drafts[draft_id]['selected_slot_index'] = selected_slot_index
                pending_drafts[draft_id]['event_id'] = event_id
                
                # Log action
                email_agent.log_action(email.message_id, 'MEETING_REQUEST', 
                                     "sent", True, True, False)
                
                return jsonify({
                    'success': True, 
                    'message': f'Meeting scheduled for {selected_slot_data["display"]} and confirmation sent!'
                })
            else:
                return jsonify({'success': False, 'message': 'Calendar event created but failed to send confirmation email'})
        
        else:
            # Handle regular draft approval
            edited_draft = data.get('edited_draft', draft_data.get('draft', ''))
            
            # Send email
            success = email_agent.send_email(
                sender_email, 
                f"Re: {email.subject}", 
                edited_draft, 
                email.thread_id
            )
            
            if success:
                # Update draft status
                pending_drafts[draft_id]['status'] = 'sent'
                
                # Log action
                was_edited = edited_draft != draft_data.get('draft', '')
                email_agent.log_action(email.message_id, draft_data['category'], 
                                     "sent", not was_edited, True, was_edited)
                
                return jsonify({'success': True, 'message': 'Email sent successfully'})
            else:
                return jsonify({'success': False, 'message': 'Failed to send email'})
            
    except Exception as e:
        logger.error(f"Error approving draft: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/email/drafts/<draft_id>/reject', methods=['POST'])
def reject_draft(draft_id):
    """Reject a draft email"""
    global pending_drafts, email_agent
    
    if draft_id not in pending_drafts:
        return jsonify({'success': False, 'message': 'Draft not found'})
    
    draft_data = pending_drafts[draft_id]
    if draft_data['status'] != 'pending':
        return jsonify({'success': False, 'message': 'Draft already processed'})
    
    try:
        # Update draft status
        pending_drafts[draft_id]['status'] = 'rejected'
        
        # Log action
        email = draft_data['email']
        if draft_data['category'] == 'MEETING_REQUEST':
            email_agent.log_action(email.message_id, 'MEETING_REQUEST', 
                                 "cancelled", False, True, False)
        else:
            email_agent.log_action(email.message_id, draft_data['category'], 
                                 "cancelled", False, True, False)
        
        return jsonify({'success': True, 'message': 'Draft rejected'})
        
    except Exception as e:
        logger.error(f"Error rejecting draft: {e}")
        return jsonify({'success': False, 'message': str(e)})

# Meeting Agent Routes
@app.route('/meeting')
def meeting_dashboard():
    """Meeting agent dashboard"""
    auth_status = check_auth()
    if not auth_status:
        return redirect(url_for('auth_page'))
    
    # Initialize agents if not already done
    if not meeting_agent:
        initialize_agents()
    
    return render_template('meeting.html', auth_status=auth_status, meeting_available=meeting_agent is not None)

@app.route('/api/meeting/recent', methods=['GET'])
def get_recent_meetings():
    """Get recent meetings"""
    global meeting_agent
    
    if not meeting_agent:
        return jsonify({'success': False, 'message': 'Meeting agent not available'})
    
    try:
        meetings = meeting_agent.list_recent_meetings()
        return jsonify({'success': True, 'meetings': meetings})
    except Exception as e:
        logger.error(f"Failed to get recent meetings: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/meeting/process', methods=['POST'])
def process_meeting():
    """Process a selected meeting"""
    global meeting_agent, meeting_processing_status
    
    if not meeting_agent:
        return jsonify({'success': False, 'message': 'Meeting agent not available'})
    
    try:
        data = request.get_json()
        meeting_index = data.get('meeting_index')
        
        if meeting_index is None:
            return jsonify({'success': False, 'message': 'Meeting index required'})
        
        # Frontend sends 0-based index (JavaScript array index)
        meeting_index = int(meeting_index)
        
        # Initialize processing status
        meeting_processing_status.update({
            'active': True,
            'current_step': 'Starting',
            'progress': {'current': 0, 'total': 6},
            'logs': [],
            'result': None
        })
        
        # Progress callback to update global status
        def progress_callback(step, current, total, message):
            meeting_processing_status.update({
                'current_step': step,
                'progress': {'current': current, 'total': total},
                'logs': meeting_processing_status['logs'] + [{
                    'timestamp': datetime.now().isoformat(),
                    'message': message,
                    'step': step,
                    'progress': f"{current}/{total}"
                }]
            })
        
        # Process meeting using web-compatible method with progress tracking
        def process_meeting_async():
            try:
                result = meeting_agent.process_meeting_by_index(meeting_index, progress_callback)
                meeting_processing_status.update({
                    'active': False,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Meeting processing error: {e}")
                meeting_processing_status.update({
                    'active': False,
                    'current_step': 'Error',
                    'logs': meeting_processing_status['logs'] + [{
                        'timestamp': datetime.now().isoformat(),
                        'message': f"❌ Error: {str(e)}",
                        'step': 'Error',
                        'progress': 'Failed'
                    }]
                })
        
        thread = threading.Thread(target=process_meeting_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Meeting processing started'})
        
    except Exception as e:
        logger.error(f"Meeting processing error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/meeting/status', methods=['GET'])
def meeting_processing_status_endpoint():
    """Get current meeting processing status"""
    global meeting_processing_status
    return jsonify({'success': True, 'status': meeting_processing_status})

# Contract Agent Routes
@app.route('/contract')
def contract_dashboard():
    """Contract agent dashboard"""
    auth_status = check_auth()
    if not auth_status:
        return redirect(url_for('auth_page'))
    
    # Initialize agents if not already done
    if not contract_agent:
        initialize_agents()
    
    return render_template('contract.html', auth_status=auth_status, contract_available=contract_agent is not None)

def analyze_contract_with_progress(filepath):
    """Analyze contract with progress tracking"""
    global contract_agent, contract_processing_status
    
    def update_progress(step, current, total, message):
        contract_processing_status['current_step'] = step
        contract_processing_status['progress'] = {'current': current, 'total': total}
        contract_processing_status['logs'].append(f"[{current}/{total}] {message}")
        print(f"[{current}/{total}] {message}")
    
    try:
        update_progress("Starting", 1, 6, "🔍 Starting contract analysis...")
        
        update_progress("Extracting Text", 2, 6, "📄 Extracting text from document...")
        # Note: The actual extraction happens inside analyze_contract
        
        update_progress("Segmenting Clauses", 3, 6, "✂️ Segmenting document into clauses using AI...")
        
        update_progress("Analyzing Content", 4, 6, "🧠 Analyzing clause content and types...")
        
        update_progress("Risk Assessment", 5, 6, "⚖️ Performing risk assessment...")
        
        # Perform the actual analysis
        analysis = contract_agent.analyze_contract(filepath)
        
        update_progress("Generating Reports", 6, 6, "📊 Generating HTML and PDF reports...")
        
        # Generate reports
        html_report_path = contract_agent.generate_report(analysis, 'html')
        pdf_report_path = contract_agent.generate_report(analysis, 'pdf')
        
        # Prepare the result
        result = {
            'success': True, 
            'message': 'Contract analyzed successfully',
            'analysis': {
                'document_id': analysis.document_id,
                'total_clauses': len(analysis.clauses),
                'high_risk_clauses': len([c for c in analysis.clauses if c.risk_assessment and c.risk_assessment.risk_level == 'HIGH']),
                'medium_risk_clauses': len([c for c in analysis.clauses if c.risk_assessment and c.risk_assessment.risk_level == 'MEDIUM']),
                'low_risk_clauses': len([c for c in analysis.clauses if c.risk_assessment and c.risk_assessment.risk_level == 'LOW']),
                'executive_summary': analysis.executive_summary,
                'processing_stats': analysis.processing_stats,
                'html_report': os.path.basename(html_report_path),
                'pdf_report': os.path.basename(pdf_report_path) if pdf_report_path else None,
                'clauses': [
                    {
                        'id': i,
                        'text': clause.text,
                        'clause_type': clause.metadata.clause_type,
                        'risk_level': clause.risk_assessment.risk_level if clause.risk_assessment else 'UNKNOWN',
                        'confidence': clause.risk_assessment.confidence if clause.risk_assessment else 0,
                        'rationale': clause.risk_assessment.rationale if clause.risk_assessment else 'No analysis available',
                        'suggestions': clause.risk_assessment.suggestions if clause.risk_assessment else [],
                        'rule_matches': clause.risk_assessment.rule_matches if clause.risk_assessment else [],
                        'priority_score': clause.risk_assessment.priority_score if clause.risk_assessment else 0
                    }
                    for i, clause in enumerate(analysis.clauses)
                ]
            }
        }
        
        contract_processing_status['result'] = result
        contract_processing_status['active'] = False
        update_progress("Complete", 6, 6, "✅ Contract analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Contract analysis error: {e}")
        contract_processing_status['result'] = {'success': False, 'message': str(e)}
        contract_processing_status['active'] = False
        update_progress("Error", 6, 6, f"❌ Analysis failed: {str(e)}")

@app.route('/api/contract/upload', methods=['POST'])
def upload_contract():
    """Upload and start contract analysis"""
    global contract_agent, contract_processing_status
    
    if not contract_agent:
        return jsonify({'success': False, 'message': 'Contract agent not available'})
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if contract_processing_status['active']:
        return jsonify({'success': False, 'message': 'Another contract is currently being processed'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Initialize processing status
            contract_processing_status = {
                'active': True,
                'current_step': 'Uploading',
                'progress': {'current': 0, 'total': 6},
                'logs': [],
                'result': None
            }
            
            # Start processing in background thread
            import threading
            thread = threading.Thread(target=analyze_contract_with_progress, args=(filepath,))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True, 
                'message': 'Contract upload successful. Analysis started.',
                'processing': True
            })
            
        except Exception as e:
            logger.error(f"Contract upload error: {e}")
            return jsonify({'success': False, 'message': str(e)})
    else:
        return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/api/contract/status', methods=['GET'])
def contract_processing_status_endpoint():
    """Get contract processing status"""
    global contract_processing_status
    return jsonify({'success': True, 'status': contract_processing_status})

@app.route('/api/contract/feedback', methods=['POST'])
def submit_clause_feedback():
    """Submit user feedback for a specific clause"""
    global contract_agent
    
    if not contract_agent:
        return jsonify({'success': False, 'message': 'Contract agent not available'})
    
    try:
        data = request.get_json()
        document_id = data.get('document_id')
        clause_id = data.get('clause_id')
        feedback_type = data.get('feedback_type')  # 'agree', 'disagree', 'suggestion'
        feedback_text = data.get('feedback_text', '')
        risk_override = data.get('risk_override')  # Optional: 'HIGH', 'MEDIUM', 'LOW'
        
        if not all([document_id, clause_id is not None, feedback_type]):
            return jsonify({'success': False, 'message': 'Missing required fields'})
        
        # Store feedback for future learning (vector database integration)
        feedback_data = {
            'document_id': document_id,
            'clause_id': clause_id,
            'feedback_type': feedback_type,
            'feedback_text': feedback_text,
            'risk_override': risk_override,
            'timestamp': datetime.now().isoformat(),
            'user_id': 'web_user'  # In a real app, this would be the authenticated user
        }
        
        # Store in contract agent's learning system
        success = contract_agent.store_user_feedback(feedback_data)
        
        if success:
            return jsonify({
                'success': True, 
                'message': 'Feedback submitted successfully. This will help improve future analysis.'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Failed to store feedback'
            })
            
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/contract/reports')
def list_contract_reports():
    """List available contract reports"""
    try:
        reports_dir = 'contractDATAtemp/reports'
        if not os.path.exists(reports_dir):
            return jsonify({'success': True, 'reports': []})
        
        reports = []
        for filename in os.listdir(reports_dir):
            if filename.endswith('.html'):
                filepath = os.path.join(reports_dir, filename)
                stat = os.stat(filepath)
                reports.append({
                    'filename': filename,
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'size': stat.st_size
                })
        
        reports.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({'success': True, 'reports': reports})
        
    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/contract/report/<filename>')
def get_contract_report(filename):
    """Get a specific contract report"""
    try:
        reports_dir = 'contractDATAtemp/reports'
        return send_from_directory(reports_dir, filename)
    except Exception as e:
        logger.error(f"Failed to get report: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("🚀 AgenticSuite Web Interface Starting...")
    print("📱 Access at: http://localhost:5000")
    
    # Initialize agents if already authenticated
    if check_auth():
        initialize_agents()
        print("✅ Agents initialized (authentication valid)")
        print("🎯 Go to http://localhost:5000 to access the dashboard")
    else:
        print("⚠️ Not authenticated - will redirect to /auth for setup")
    
    app.run(debug=True, host='0.0.0.0', port=5000)