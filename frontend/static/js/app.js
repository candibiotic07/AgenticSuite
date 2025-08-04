/**
 * AgenticSuite - Frontend JavaScript
 * Handles API communications and UI interactions
 */

// Global state
window.AgenticSuite = {
    baseURL: '',
    authStatus: false,
    notifications: [],
    
    // Initialize the application
    init() {
        this.checkAuthStatus();
        this.setupGlobalEventListeners();
        this.loadUserPreferences();
    },
    
    // Check authentication status
    async checkAuthStatus() {
        try {
            const response = await fetch('/api/auth/status');
            const data = await response.json();
            this.authStatus = data.authenticated;
            this.updateAuthUI();
        } catch (error) {
            console.error('Auth status check failed:', error);
            this.authStatus = false;
            this.updateAuthUI();
        }
    },
    
    // Update authentication UI elements
    updateAuthUI() {
        const authElements = document.querySelectorAll('[data-auth-required]');
        authElements.forEach(element => {
            if (this.authStatus) {
                element.style.display = 'block';
            } else {
                element.style.display = 'none';
            }
        });
        
        const noAuthElements = document.querySelectorAll('[data-no-auth]');
        noAuthElements.forEach(element => {
            if (!this.authStatus) {
                element.style.display = 'block';
            } else {
                element.style.display = 'none';
            }
        });
    },
    
    // Setup global event listeners
    setupGlobalEventListeners() {
        // Close alerts
        document.addEventListener('click', (e) => {
            if (e.target.matches('.alert-close') || e.target.matches('.alert-close *')) {
                const alert = e.target.closest('.alert');
                if (alert) {
                    alert.remove();
                }
            }
        });
        
        // Handle form submissions
        document.addEventListener('submit', (e) => {
            if (e.target.matches('[data-api-form]')) {
                e.preventDefault();
                this.handleApiForm(e.target);
            }
        });
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'k':
                        e.preventDefault();
                        // Focus search if available
                        const searchInput = document.querySelector('input[type="search"]');
                        if (searchInput) searchInput.focus();
                        break;
                }
            }
        });
    },
    
    // Load user preferences from localStorage
    loadUserPreferences() {
        const prefs = localStorage.getItem('agenticsuite_prefs');
        if (prefs) {
            try {
                const preferences = JSON.parse(prefs);
                this.applyPreferences(preferences);
            } catch (error) {
                console.error('Failed to load preferences:', error);
            }
        }
    },
    
    // Apply user preferences
    applyPreferences(preferences) {
        // Theme
        if (preferences.theme) {
            document.documentElement.setAttribute('data-theme', preferences.theme);
        }
        
        // UI preferences
        if (preferences.compactMode) {
            document.documentElement.classList.toggle('compact-mode', preferences.compactMode);
        }
    },
    
    // Save user preferences
    savePreferences(preferences) {
        try {
            localStorage.setItem('agenticsuite_prefs', JSON.stringify(preferences));
        } catch (error) {
            console.error('Failed to save preferences:', error);
        }
    }
};

// Notification system
window.AgenticSuite.notifications = {
    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
        
        return notification;
    },
    
    success(message, duration) {
        return this.show(message, 'success', duration);
    },
    
    error(message, duration) {
        return this.show(message, 'error', duration);
    },
    
    warning(message, duration) {
        return this.show(message, 'warning', duration);
    }
};

// API utilities
window.AgenticSuite.api = {
    async request(url, options = {}) {
        const defaults = {
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        const config = Object.assign({}, defaults, options);
        
        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    async get(url) {
        return this.request(url, { method: 'GET' });
    },
    
    async post(url, data) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    async put(url, data) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    async delete(url) {
        return this.request(url, { method: 'DELETE' });
    },
    
    // Upload file with progress
    async uploadFile(url, file, onProgress) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);
            
            const xhr = new XMLHttpRequest();
            
            if (onProgress) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });
            }
            
            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (error) {
                        resolve(xhr.responseText);
                    }
                } else {
                    reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
                }
            };
            
            xhr.onerror = function() {
                reject(new Error('Upload failed: Network error'));
            };
            
            xhr.open('POST', url);
            xhr.send(formData);
        });
    }
};

// Loading utilities
window.AgenticSuite.loading = {
    show(text = 'Loading...') {
        const modal = document.getElementById('loadingModal');
        const loadingText = document.getElementById('loadingText');
        
        if (modal && loadingText) {
            loadingText.textContent = text;
            modal.style.display = 'flex';
        }
    },
    
    hide() {
        const modal = document.getElementById('loadingModal');
        if (modal) {
            modal.style.display = 'none';
        }
    },
    
    // Wrap an async function with loading indicator
    async wrap(asyncFn, loadingText) {
        this.show(loadingText);
        try {
            return await asyncFn();
        } finally {
            this.hide();
        }
    }
};

// Form utilities
window.AgenticSuite.forms = {
    // Serialize form data to object
    serialize(form) {
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            if (data[key]) {
                // Handle multiple values (convert to array)
                if (Array.isArray(data[key])) {
                    data[key].push(value);
                } else {
                    data[key] = [data[key], value];
                }
            } else {
                data[key] = value;
            }
        }
        
        return data;
    },
    
    // Validate form fields
    validate(form) {
        const errors = [];
        const requiredFields = form.querySelectorAll('[required]');
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                errors.push(`${field.name || field.id || 'Field'} is required`);
                field.classList.add('error');
            } else {
                field.classList.remove('error');
            }
        });
        
        // Email validation
        const emailFields = form.querySelectorAll('input[type="email"]');
        emailFields.forEach(field => {
            if (field.value && !this.isValidEmail(field.value)) {
                errors.push(`${field.name || field.id || 'Email'} is not valid`);
                field.classList.add('error');
            }
        });
        
        return errors;
    },
    
    // Email validation regex
    isValidEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    },
    
    // Reset form with visual feedback
    reset(form) {
        form.reset();
        form.querySelectorAll('.error').forEach(field => {
            field.classList.remove('error');
        });
    }
};

// Utility functions
window.AgenticSuite.utils = {
    // Format file size
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Format date/time
    formatDateTime(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString();
    },
    
    // Format relative time (e.g., "2 hours ago")
    formatRelativeTime(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        
        const diffSeconds = Math.floor(diffMs / 1000);
        const diffMinutes = Math.floor(diffSeconds / 60);
        const diffHours = Math.floor(diffMinutes / 60);
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffDays > 0) {
            return diffDays === 1 ? 'Yesterday' : `${diffDays} days ago`;
        } else if (diffHours > 0) {
            return diffHours === 1 ? '1 hour ago' : `${diffHours} hours ago`;
        } else if (diffMinutes > 0) {
            return diffMinutes === 1 ? '1 minute ago' : `${diffMinutes} minutes ago`;
        } else {
            return 'Just now';
        }
    },
    
    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Throttle function
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Copy text to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            window.AgenticSuite.notifications.success('Copied to clipboard');
        } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            window.AgenticSuite.notifications.error('Failed to copy to clipboard');
        }
    },
    
    // Generate unique ID
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
};

// Agent-specific functionality
window.AgenticSuite.agents = {
    email: {
        async processEmails(config = {}) {
            try {
                const response = await window.AgenticSuite.api.post('/api/email/process', config);
                if (response.success) {
                    window.AgenticSuite.notifications.success('Email processing started');
                    return response;
                } else {
                    throw new Error(response.message);
                }
            } catch (error) {
                window.AgenticSuite.notifications.error('Failed to process emails: ' + error.message);
                throw error;
            }
        },
        
        async getMetrics() {
            try {
                const response = await window.AgenticSuite.api.get('/api/email/metrics');
                return response.success ? response.metrics : {};
            } catch (error) {
                console.error('Failed to get email metrics:', error);
                return {};
            }
        }
    },
    
    meeting: {
        async getRecentMeetings() {
            try {
                const response = await window.AgenticSuite.api.get('/api/meeting/recent');
                if (response.success) {
                    return response.meetings;
                } else {
                    throw new Error(response.message);
                }
            } catch (error) {
                window.AgenticSuite.notifications.error('Failed to load meetings: ' + error.message);
                throw error;
            }
        },
        
        async processMeeting(meetingIndex) {
            try {
                const response = await window.AgenticSuite.api.post('/api/meeting/process', {
                    meeting_index: meetingIndex
                });
                if (response.success) {
                    window.AgenticSuite.notifications.success('Meeting processing started');
                    return response;
                } else {
                    throw new Error(response.message);
                }
            } catch (error) {
                window.AgenticSuite.notifications.error('Failed to process meeting: ' + error.message);
                throw error;
            }
        }
    },
    
    contract: {
        async uploadAndAnalyze(file, onProgress) {
            try {
                const response = await window.AgenticSuite.api.uploadFile(
                    '/api/contract/upload',
                    file,
                    onProgress
                );
                if (response.success) {
                    window.AgenticSuite.notifications.success('Contract analyzed successfully');
                    return response;
                } else {
                    throw new Error(response.message);
                }
            } catch (error) {
                window.AgenticSuite.notifications.error('Failed to analyze contract: ' + error.message);
                throw error;
            }
        },
        
        async getReports() {
            try {
                const response = await window.AgenticSuite.api.get('/api/contract/reports');
                return response.success ? response.reports : [];
            } catch (error) {
                console.error('Failed to get contract reports:', error);
                return [];
            }
        }
    }
};

// Keyboard shortcuts
window.AgenticSuite.shortcuts = {
    init() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in inputs
            if (e.target.matches('input, textarea, select')) {
                return;
            }
            
            switch (e.key) {
                case 'h':
                    // Go to home/dashboard
                    if (!e.ctrlKey && !e.metaKey) {
                        window.location.href = '/';
                    }
                    break;
                case 'e':
                    // Go to email agent
                    if (!e.ctrlKey && !e.metaKey) {
                        window.location.href = '/email';
                    }
                    break;
                case 'm':
                    // Go to meeting agent
                    if (!e.ctrlKey && !e.metaKey) {
                        window.location.href = '/meeting';
                    }
                    break;
                case 'c':
                    // Go to contract agent
                    if (!e.ctrlKey && !e.metaKey) {
                        window.location.href = '/contract';
                    }
                    break;
                case '?':
                    // Show help
                    this.showHelp();
                    break;
            }
        });
    },
    
    showHelp() {
        const helpContent = `
            <div style="text-align: left;">
                <h3>Keyboard Shortcuts</h3>
                <ul style="list-style: none; padding: 0;">
                    <li><kbd>h</kbd> - Go to Dashboard</li>
                    <li><kbd>e</kbd> - Go to Email Agent</li>
                    <li><kbd>m</kbd> - Go to Meeting Agent</li>
                    <li><kbd>c</kbd> - Go to Contract Agent</li>
                    <li><kbd>Ctrl/Cmd + K</kbd> - Focus search</li>
                    <li><kbd>?</kbd> - Show this help</li>
                </ul>
            </div>
        `;
        
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.style.display = 'flex';
        modal.innerHTML = `
            <div class="modal-content">
                ${helpContent}
                <button onclick="this.closest('.modal').remove()" class="btn btn-primary" style="margin-top: 1rem;">
                    Close
                </button>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close on escape
        const closeOnEscape = (e) => {
            if (e.key === 'Escape') {
                modal.remove();
                document.removeEventListener('keydown', closeOnEscape);
            }
        };
        document.addEventListener('keydown', closeOnEscape);
    }
};

// Auto-refresh functionality
window.AgenticSuite.autoRefresh = {
    intervals: new Map(),
    
    start(key, callback, intervalMs = 30000) {
        this.stop(key); // Clear existing interval
        const interval = setInterval(callback, intervalMs);
        this.intervals.set(key, interval);
    },
    
    stop(key) {
        const interval = this.intervals.get(key);
        if (interval) {
            clearInterval(interval);
            this.intervals.delete(key);
        }
    },
    
    stopAll() {
        this.intervals.forEach((interval, key) => {
            clearInterval(interval);
        });
        this.intervals.clear();
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.AgenticSuite.init();
    window.AgenticSuite.shortcuts.init();
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    window.AgenticSuite.autoRefresh.stopAll();
});

// Global error handler
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
    
    // Don't show notifications for script loading errors
    if (e.filename && e.filename.includes('.js')) {
        return;
    }
    
    window.AgenticSuite.notifications.error('An unexpected error occurred');
});

// Global promise rejection handler
window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    
    // Prevent default browser behavior
    e.preventDefault();
    
    window.AgenticSuite.notifications.error('An unexpected error occurred');
});

// Export for use in other scripts
window.showNotification = window.AgenticSuite.notifications.show.bind(window.AgenticSuite.notifications);