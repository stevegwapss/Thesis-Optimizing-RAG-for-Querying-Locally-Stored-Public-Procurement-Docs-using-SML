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
        console.log('Document viewer init called');
        
        const fileName = sessionStorage.getItem('uploadedPdf');
        const pdfDataStr = sessionStorage.getItem('pdfData');
        const uploadedFolder = sessionStorage.getItem('uploadedFolder');
        const folderUploadResult = sessionStorage.getItem('folderUploadResult');
        
        console.log('Session storage values:', {
            fileName,
            pdfDataStr: pdfDataStr ? 'exists' : 'null',
            uploadedFolder,
            folderUploadResult: folderUploadResult ? 'exists' : 'null'
        });
        
        // Check if this is a folder upload
        if (uploadedFolder === 'true' && folderUploadResult) {
            console.log('Detected folder upload, calling handleFolderUpload');
            handleFolderUpload(JSON.parse(folderUploadResult));
            return;
        }
        
        if (!fileName) {
            console.log('No fileName found, redirecting to index.html');
            // No document found, redirect back to upload page
            window.location.href = 'index.html';
            return;
        }
        
        console.log('Processing single file upload for:', fileName);
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
    
    // Handle folder upload results
    async function handleFolderUpload(folderResult) {
        console.log('handleFolderUpload called with:', folderResult);
        
        // Get real-time database stats instead of using cached values
        try {
            const response = await fetch('/database-stats');
            const stats = await response.json();
            
            if (stats.success) {
                // Use real-time stats instead of cached folderResult
                const actualFileCount = stats.unique_files;
                console.log('Real-time file count:', actualFileCount);
                
                // Update document name with actual count
                const folderName = folderResult.folder_path ? folderResult.folder_path.split('/').pop() || folderResult.folder_path.split('\\').pop() : 'Processed Folder';
                documentName.textContent = `📁 ${folderName} (${actualFileCount} files processed)`;
                
                // Update folderResult with real data
                folderResult.processed_files = actualFileCount;
                folderResult.successful_files = actualFileCount;
                folderResult.total_files = actualFileCount;
            }
        } catch (error) {
            console.error('Error fetching database stats:', error);
            // Fall back to cached values if request fails
        }
        
        console.log('Document name updated to:', documentName.textContent);
        
        // Replace PDF container content with folder summary
        showFolderSummary(folderResult);
        
        // Initialize chat for querying across all documents
        initChat();
        
        // Check Ollama connection status
        checkStatus();
    }
    
    // Show folder upload summary with file previews
    async function showFolderSummary(folderResult) {
        const pdfContainer = document.getElementById('pdf-container');
        if (!pdfContainer) return;
        
        // Fetch detailed file information from the database
        try {
            const response = await fetch('/database-stats');
            const stats = await response.json();
            
            if (stats.success && stats.file_details) {
                showFileList(stats.file_details);
            } else {
                // Fallback to basic summary if detailed info not available
                showBasicSummary(folderResult);
            }
        } catch (error) {
            console.error('Error fetching file details:', error);
            showBasicSummary(folderResult);
        }
    }
    
    // Show list of uploaded files with preview functionality
    function showFileList(fileDetails) {
        const pdfContainer = document.getElementById('pdf-container');
        if (!pdfContainer) return;
        
        const fileList = Object.values(fileDetails);
        
        pdfContainer.innerHTML = `
            <div class="file-list">
                <div class="file-list-header">
                    <h3>📁 Uploaded Documents</h3>
                    <p>Click on any file to preview it</p>
                </div>
                <div class="file-items">
                    ${fileList.map((file, index) => `
                        <div class="file-item ${index === 0 ? 'active' : ''}" data-file-url="${file.upload_url}" data-filename="${file.filename}">
                            <div class="file-icon">📄</div>
                            <div class="file-info">
                                <div class="file-name">${file.filename}</div>
                                <div class="file-status">
                                    ${file.available_in_uploads ? 
                                        '<span class="status-available">✅ Available for preview</span>' : 
                                        '<span class="status-unavailable">❌ Preview not available</span>'
                                    }
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        // Add click handlers for file items
        const fileItems = pdfContainer.querySelectorAll('.file-item');
        fileItems.forEach(item => {
            item.addEventListener('click', () => {
                const fileUrl = item.dataset.fileUrl;
                const filename = item.dataset.filename;
                
                if (fileUrl) {
                    // Remove active class from all items
                    fileItems.forEach(fi => fi.classList.remove('active'));
                    // Add active class to clicked item
                    item.classList.add('active');
                    
                    // Update document name
                    document.getElementById('document-name').textContent = filename;
                    
                    // Load the PDF
                    loadPdf(fileUrl);
                } else {
                    alert('This file is not available for preview');
                }
            });
        });
        
        // Load the first available file automatically
        const firstAvailableFile = fileList.find(file => file.available_in_uploads);
        if (firstAvailableFile) {
            document.getElementById('document-name').textContent = firstAvailableFile.filename;
            loadPdf(firstAvailableFile.upload_url);
        }
        
        // Update PDF viewer header
        const pdfViewerHeader = document.querySelector('.pdf-viewer-header h4');
        if (pdfViewerHeader) {
            pdfViewerHeader.textContent = '📄 Document Preview';
        }
    }
    
    // Fallback basic summary (original functionality)
    function showBasicSummary(folderResult) {
        const pdfContainer = document.getElementById('pdf-container');
        if (!pdfContainer) return;
        
        // Clear existing content and show summary
        pdfContainer.innerHTML = `
            <div class="folder-summary">
                <div class="summary-header">
                    <h3>📁 Batch Upload Complete</h3>
                    <p>Your documents have been processed and are ready for querying!</p>
                </div>
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-number">${folderResult.processed_files || folderResult.successful_files || 0}</div>
                        <div class="stat-label">Files Processed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${folderResult.total_files || 0}</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    ${folderResult.errors && folderResult.errors.length > 0 ? `
                    <div class="stat-item error">
                        <div class="stat-number">${folderResult.errors.length}</div>
                        <div class="stat-label">Errors</div>
                    </div>
                    ` : ''}
                </div>
                ${folderResult.errors && folderResult.errors.length > 0 ? `
                <div class="error-details">
                    <h4>⚠️ Processing Errors:</h4>
                    <ul>
                        ${folderResult.errors.map(error => `<li>${error}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
                <div class="summary-footer">
                    <p>💡 <strong>Tip:</strong> Use the chat to ask questions about any of your uploaded documents.</p>
                </div>
            </div>
        `;
        
        // Update PDF viewer header
        const pdfViewerHeader = document.querySelector('.pdf-viewer-header h4');
        if (pdfViewerHeader) {
            pdfViewerHeader.textContent = '📊 Upload Summary';
        }
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