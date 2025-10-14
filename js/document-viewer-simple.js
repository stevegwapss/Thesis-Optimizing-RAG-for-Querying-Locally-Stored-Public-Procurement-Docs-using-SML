// document-viewer-simple.js - Clean version for file listing and URL viewing

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    const fileList = document.getElementById('file-list');
    const fileCount = document.getElementById('file-count');
    const roleSelect = document.getElementById('roleSelect');
    
    // API endpoints
    const API_ENDPOINTS = {
        QUERY: 'http://127.0.0.1:5000/query',
        STATUS: 'http://127.0.0.1:5000/status',
        UPLOADS: 'http://127.0.0.1:5000/uploads'
    };

    // Initialize the simplified interface
    function init() {
        console.log('Simplified document viewer initialized');
        
        // Load available files
        loadFileList();
        
        // Set up event listeners
        setupEventListeners();
    }
    
    // Load and display available files
    async function loadFileList() {
        try {
            // Get list of uploaded files from the uploads directory
            const response = await fetch('/api/files');
            if (response.ok) {
                const files = await response.json();
                displayFileList(files);
            } else {
                // Fallback: show common file types from uploads folder
                const fallbackFiles = [
                    'Affidavit-of-Undertaking.pdf',
                    'FO-UHS-032 Patient Consultation Record (1).pdf',
                    'Medical-Waiver-Updated.pdf',
                    'Student-Waiver-Extension-Updated.pdf',
                    'Student-WaiverOfLiability-Updated.pdf'
                ];
                displayFileList(fallbackFiles);
            }
        } catch (error) {
            console.error('Error loading file list:', error);
            showError('Failed to load file list');
        }
    }
    
    // Display files in the file list panel
    function displayFileList(files) {
        if (!fileList) return;
        
        if (files.length === 0) {
            fileList.innerHTML = `
                <div class="no-files">
                    <p>No documents found</p>
                    <small>Upload documents through the main interface</small>
                </div>
            `;
            return;
        }
        
        fileList.innerHTML = files.map(filename => `
            <div class="file-item" data-filename="${filename}" onclick="openFile('${filename}')">
                <span class="file-icon">📄</span>
                <span class="file-name" title="${filename}">${filename}</span>
            </div>
        `).join('');
        
        // Update file count
        if (fileCount) {
            fileCount.textContent = `${files.length} documents available`;
        }
    }
    
    // Open file in new tab via URL
    function openFile(filename) {
        console.log('Opening file:', filename);
        
        // Highlight selected file
        document.querySelectorAll('.file-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const clickedItem = document.querySelector(`[data-filename="${filename}"]`);
        if (clickedItem) {
            clickedItem.classList.add('active');
        }
        
        // Open file in new tab
        const fileUrl = `${API_ENDPOINTS.UPLOADS}/${encodeURIComponent(filename)}`;
        window.open(fileUrl, '_blank');
    }
    
    // Set up event listeners
    function setupEventListeners() {
        // Chat functionality
        if (sendButton) {
            sendButton.addEventListener('click', sendMessage);
        }
        
        if (queryInput) {
            queryInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        }
        
        // Role selection
        if (roleSelect) {
            roleSelect.addEventListener('change', function() {
                console.log('Role changed to:', this.value);
                // Update welcome message based on role
                updateWelcomeMessage(this.value);
            });
        }
    }
    
    // Send message to RAG system
    async function sendMessage() {
        const message = queryInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        queryInput.value = '';
        
        try {
            const response = await fetch(API_ENDPOINTS.QUERY, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    role: roleSelect ? roleSelect.value : 'general'
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                addMessage(data.response || 'No response received', 'assistant');
                
                // Show sources if available
                if (data.sources && data.sources.length > 0) {
                    const sourcesText = `\n\nSources: ${data.sources.join(', ')}`;
                    addMessage(sourcesText, 'sources');
                }
            } else {
                addMessage('Sorry, there was an error processing your request.', 'assistant');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('Sorry, there was an error connecting to the server.', 'assistant');
        }
    }
    
    // Add message to chat
    function addMessage(content, type) {
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        
        if (type === 'sources') {
            messageDiv.innerHTML = `<small style="color: #666; font-style: italic;">${content}</small>`;
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Update welcome message based on role
    function updateWelcomeMessage(role) {
        const roleMessages = {
            general: "Hello! I've indexed your documents. You can ask me any questions about the content or click any document in the list to view it.",
            auditor: "Welcome, Auditor! I can help you review compliance, audit trails, and regulatory requirements in the documents.",
            procurement_officer: "Welcome, Procurement Officer! I can assist with procurement processes, vendor requirements, and contract details.",
            policy_maker: "Welcome, Policy Maker! I can help analyze policies, regulations, and strategic implications in the documents.",
            bidder: "Welcome, Bidder/Supplier! I can help you understand requirements, specifications, and submission guidelines."
        };
        
        const welcomeMsg = roleMessages[role] || roleMessages.general;
        
        // Find and update the first assistant message
        const firstAssistantMsg = chatMessages.querySelector('.assistant-message');
        if (firstAssistantMsg) {
            firstAssistantMsg.textContent = welcomeMsg;
        }
    }
    
    // Show error message
    function showError(message) {
        if (fileList) {
            fileList.innerHTML = `
                <div class="error-message" style="text-align: center; padding: 20px; color: #dc3545;">
                    <p>⚠️ ${message}</p>
                </div>
            `;
        }
    }
    
    // Make functions globally available
    window.openFile = openFile;
    window.loadFileList = loadFileList;
    
    // Initialize the application
    init();
});