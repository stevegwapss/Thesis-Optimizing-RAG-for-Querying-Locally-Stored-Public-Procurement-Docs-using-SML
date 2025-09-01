// document-viewer.js

// PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.10.377/pdf.worker.min.js';

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const documentName = document.getElementById('document-name');
    const pdfContainer = document.getElementById('pdf-container');
    const canvas = document.getElementById('pdf-render');
    const ctx = canvas.getContext('2d');
    const currentPageEl = document.getElementById('current-page');
    const totalPagesEl = document.getElementById('total-pages');
    const prevButton = document.getElementById('prev-page');
    const nextButton = document.getElementById('next-page');
    const zoomInButton = document.getElementById('zoom-in');
    const zoomOutButton = document.getElementById('zoom-out');
    const loadingIndicator = document.getElementById('pdf-loading');
    const chatMessages = document.getElementById('chat-messages');
    const queryInput = document.getElementById('query-input');
    const sendButton = document.getElementById('send-button');
    
    // PDF rendering variables
    let pdfDoc = null;
    let currentPage = 1;
    let scale = 1.2;
    let pdfData = null;
    
    // API endpoints
    const API_ENDPOINTS = {
        UPLOAD: 'http://127.0.0.1:5000/upload',
        QUERY: 'http://127.0.0.1:5000/query',
        STATUS: 'http://127.0.0.1:5000/status'
    };
    
    // Check if we have a document in session storage
    function init() {
        const fileName = sessionStorage.getItem('uploadedPdf');
        const pdfDataStr = sessionStorage.getItem('pdfData');
        
        if (!fileName) {
            // No document found, redirect back to upload page
            window.location.href = 'index.html';
            return;
        }
        
        documentName.textContent = fileName;
        
        if (pdfDataStr) {
            pdfData = JSON.parse(pdfDataStr);
            
            // Check if the file was already uploaded to the server
            if (pdfData.uploaded) {
                // Load the PDF 
                loadPdf(pdfData.url);
            } else if (pdfData.base64) {
                // Convert base64 to Uint8Array for PDF.js
                const binaryString = atob(pdfData.base64.split(',')[1]);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                // Upload to server first
                uploadPdfToServer(fileName, bytes, function() {
                    // Then load it
                    loadPdf(bytes);
                });
            } else {
                // Fallback to sample PDF
                loadSamplePdf();
            }
        } else {
            // Fallback to sample PDF
            loadSamplePdf();
        }
        
        // Initialize chat
        initChat();
        
        // Check Ollama connection status
        checkStatus();
    }
    
    // Load sample PDF (fallback)
    function loadPdf(source) {
        loadingIndicator.style.display = 'flex';
        
        let pdfSource = source;
        
        // If source is a string that looks like a filename rather than a URL or data
        if (typeof source === 'string' && !source.startsWith('http') && !source.startsWith('data:')) {
            // Construct path to the uploaded file
            pdfSource = `http://127.0.0.1:5000/uploads/${source}`;
        }
        
        // Source can be URL string or Uint8Array
        pdfjsLib.getDocument(pdfSource).promise.then(function(pdf) {
            pdfDoc = pdf;
            totalPagesEl.textContent = pdf.numPages;
            
            // Render first page
            renderPage(currentPage);
            
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
        }).catch(function(error) {
            console.error('Error loading PDF:', error);
            loadingIndicator.style.display = 'none';
            alert('Failed to load PDF document. Please try again.');
        });
    }
    
    // Upload PDF to server
    function uploadPdfToServer(fileName, pdfBytes, callback) {
        // Create form data
        const formData = new FormData();
        const blob = new Blob([pdfBytes], { type: 'application/pdf' });
        formData.append('file', blob, fileName);
        
        // Show loading
        loadingIndicator.style.display = 'flex';
        
        // Upload to server
        fetch(API_ENDPOINTS.UPLOAD, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update session storage
                const pdfData = {
                    uploaded: true,
                    fileName: fileName
                };
                sessionStorage.setItem('pdfData', JSON.stringify(pdfData));
                
                // Call callback
                if (callback) callback();
            } else {
                console.error('Upload failed:', data.message);
                alert('Failed to upload document: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Upload error:', error);
            alert('Failed to upload document. Please try again.');
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
        });
    }
    
    // Check status of Ollama connection
    function checkStatus() {
        fetch(API_ENDPOINTS.STATUS)
        .then(response => response.json())
        .then(data => {
            if (data.ollama_status !== 'connected') {
                addMessage('system', 'Warning: Ollama service is not connected. Please ensure Ollama is running.');
            }
        })
        .catch(error => {
            console.error('Status check error:', error);
        });
    }
    
    // Load PDF document
    function loadPdf(source) {
        loadingIndicator.style.display = 'flex';
        
        // Source can be URL string or Uint8Array
        pdfjsLib.getDocument(source).promise.then(function(pdf) {
            pdfDoc = pdf;
            totalPagesEl.textContent = pdf.numPages;
            
            // Render first page
            renderPage(currentPage);
        }).catch(function(error) {
            console.error('Error loading PDF:', error);
            loadingIndicator.style.display = 'none';
            alert('Failed to load PDF document.');
        });
    }
    
    // Render a specific page
    function renderPage(pageNumber) {
        loadingIndicator.style.display = 'flex';
        
        pdfDoc.getPage(pageNumber).then(function(page) {
            const viewport = page.getViewport({ scale });
            
            // Set canvas dimensions
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            
            // Render PDF page
            const renderContext = {
                canvasContext: ctx,
                viewport: viewport
            };
            
            page.render(renderContext).promise.then(function() {
                loadingIndicator.style.display = 'none';
                currentPageEl.textContent = pageNumber;
                
                // Update button states
                prevButton.disabled = pageNumber <= 1;
                nextButton.disabled = pageNumber >= pdfDoc.numPages;
            });
        });
    }
    
    // Navigate to previous page
    function goPrevPage() {
        if (currentPage > 1) {
            currentPage--;
            renderPage(currentPage);
        }
    }
    
    // Navigate to next page
    function goNextPage() {
        if (currentPage < pdfDoc.numPages) {
            currentPage++;
            renderPage(currentPage);
        }
    }
    
    // Zoom in
    function zoomIn() {
        scale += 0.2;
        renderPage(currentPage);
    }
    
    // Zoom out
    function zoomOut() {
        if (scale > 0.4) {
            scale -= 0.2;
            renderPage(currentPage);
        }
    }
    
    // Initialize chat interface
    function initChat() {
        // Add event listeners for chat interface
        queryInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendQuery();
            }
        });
        
        sendButton.addEventListener('click', sendQuery);
        
        // Add role selection change handler
        const roleSelect = document.getElementById('roleSelect');
        if (roleSelect) {
            roleSelect.addEventListener('change', function() {
                const newRole = this.value;
                const roleDisplayNames = {
                    'general': 'General User',
                    'auditor': 'Auditor',
                    'procurement_officer': 'Procurement Officer',
                    'policy_maker': 'Policy Maker',
                    'bidder': 'Bidder/Supplier'
                };
                
                const displayName = roleDisplayNames[newRole] || newRole;
                
                // Add a system message to show role change
                addMessage('system', `🎭 Role switched to: ${displayName}`);
                
                console.log(`Role changed to: ${newRole}`);
            });
        }
        
        // Add welcome message with initial role
        const initialRole = roleSelect ? roleSelect.value : 'general';
        const roleDisplayNames = {
            'general': 'General User',
            'auditor': 'Auditor',
            'procurement_officer': 'Procurement Officer',
            'policy_maker': 'Policy Maker',
            'bidder': 'Bidder/Supplier'
        };
        
        addMessage('system', `Welcome! Current role: ${roleDisplayNames[initialRole] || initialRole}. You can ask questions about the document content.`);
    }
    
    // Send query to AI
    function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Get selected role from dropdown - more robust selection
        const roleSelect = document.getElementById('roleSelect');
        let selectedRole = 'general';
        
        if (roleSelect && roleSelect.value) {
            selectedRole = roleSelect.value;
        }
        
        console.log(`DEBUG: Selected role from dropdown: "${selectedRole}"`); // Debug logging
        
        // Add user message to chat (with role indicator for clarity)
        addMessage('user', `[${selectedRole.toUpperCase()}] ${query}`);
        
        // Clear input
        queryInput.value = '';
        
        // Show thinking indicator
        showThinking();
        
        // Send query to backend with role
        fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                query: query,
                role: selectedRole  // Send the role
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove thinking message
            removeThinking();
            
            console.log('Response received:', data); // Debug log
            
            if (data.error) {
                addMessage('system', `Error: ${data.error}`);
            } else {
                // Add response with role confirmation and metadata
                const roleInfo = data.confirmed_role ? `[${data.confirmed_role.toUpperCase()}]` : '';
                const tagsInfo = data.relevant_tags && data.relevant_tags.length > 0 ? 
                    ` | Tags: ${data.relevant_tags.join(', ')}` : '';
                const contextInfo = ` | Chunks: ${data.contexts_used || 0}`;
                
                // Add metadata line for transparency
                addMessage('system', `${roleInfo} Response${tagsInfo}${contextInfo}`);
                
                // Add the actual response
                addMessage('assistant', data.response);
            }
        })
        .catch(error => {
            removeThinking();
            console.error('Query error:', error);
            addMessage('system', 'Failed to get a response. Please try again.');
        });
    }

    // Remove the addEnhancedMessage function and keep only the simple addMessage
    function addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        messageDiv.textContent = content;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Show thinking message
    function showThinking() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'assistant-message thinking-message';
        thinkingDiv.innerHTML = '<div class="spinner" style="width: 20px; height: 20px; margin-right: 10px;"></div> Thinking...';
        chatMessages.appendChild(thinkingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Remove thinking message
    function removeThinking() {
        const thinkingMsg = document.querySelector('.thinking-message');
        if (thinkingMsg) {
            thinkingMsg.remove();
        }
    }
    
    // Add message to chat
    function addMessage(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        messageDiv.textContent = content;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Add event listeners
    prevButton.addEventListener('click', goPrevPage);
    nextButton.addEventListener('click', goNextPage);
    zoomInButton.addEventListener('click', zoomIn);
    zoomOutButton.addEventListener('click', zoomOut);
    
    // Initialize
    init();
});