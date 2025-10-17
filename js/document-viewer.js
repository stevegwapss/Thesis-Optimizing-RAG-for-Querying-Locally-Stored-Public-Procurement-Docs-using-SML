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
    const chatMessages = document.getElementById('messages');
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
        
        // Check if required DOM elements exist
        if (!documentName) {
            console.error('documentName element not found');
            return;
        }
        
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
            console.log('No fileName found, trying to load available files');
            // Try to load files from the server
            loadAvailableFiles();
            return;
        }
        
        console.log('Processing single file upload for:', fileName);
        if (documentName) {
            documentName.textContent = fileName;
        }
        
        // Always try to load the file from the server first
        const serverUrl = `http://127.0.0.1:5000/uploads/${fileName}`;
        console.log('Trying to load PDF from server:', serverUrl);
        
        loadPdfFromServer(serverUrl, fileName);
        
        // Initialize chat
        initChat();
    }
    
    // Load available files from server
    function loadAvailableFiles() {
        console.log('Loading available files from server...');
        
        fetch('http://127.0.0.1:5000/files')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.files && data.files.length > 0) {
                console.log('Found available files:', data.files);
                
                // Set the first available file as the document name
                const firstFile = data.files[0];
                if (documentName) {
                    documentName.textContent = firstFile;
                }
                
                // Try to load the first available PDF
                const serverUrl = `http://127.0.0.1:5000/uploads/${firstFile}`;
                loadPdfFromServer(serverUrl, firstFile);
                
                // Show file list in PDF viewer
                showFileListInViewer(data.file_statistics);
            } else {
                console.log('No files available, showing upload message');
                showNoFilesMessage();
            }
        })
        .catch(error => {
            console.error('Error loading files:', error);
            showNoFilesMessage();
        });
        
        // Initialize chat anyway
        initChat();
    }
    
    // Load PDF from server URL
    function loadPdfFromServer(url, filename) {
        console.log('Loading PDF from URL:', url);
        const loadingIndicator = document.getElementById('pdf-loading');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'flex';
        }
        
        pdfjsLib.getDocument(url).promise.then(function(pdf) {
            console.log('PDF loaded successfully:', filename);
            pdfDoc = pdf;
            
            // Update page controls
            if (totalPagesEl) {
                totalPagesEl.textContent = pdf.numPages;
            }
            currentPage = 1;
            if (currentPageEl) {
                currentPageEl.textContent = currentPage;
            }
            
            // Update navigation button states
            if (prevButton) {
                prevButton.disabled = currentPage <= 1;
            }
            if (nextButton) {
                nextButton.disabled = currentPage >= pdf.numPages;
            }
            
            // Render first page
            renderPage(currentPage);
        }).catch(function(error) {
            console.error('Error loading PDF from server:', error);
            console.log('Trying fallback loading method...');
            
            // Fallback: try to load as direct file
            loadPdfFallback(filename);
        })
        .finally(() => {
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
        });
    }
    
    // Fallback PDF loading
    function loadPdfFallback(filename) {
        console.log('Using fallback loading for:', filename);
        
        // Show message instead of trying to load placeholder
        const pdfContainer = document.getElementById('pdf-container');
        if (pdfContainer) {
            pdfContainer.innerHTML = `
                <div class="pdf-error">
                    <div class="error-icon">📄</div>
                    <h3>Document: ${filename}</h3>
                    <p>Preview not available</p>
                    <p>The document has been processed and is available for querying.</p>
                    <button onclick="location.reload()" class="reload-btn">Reload Page</button>
                </div>
            `;
        }
    }
    
    // Show file list in viewer with PDF thumbnails
    function showFileListInViewer(fileStats) {
        const pdfContainer = document.getElementById('pdf-container');
        if (!pdfContainer || !fileStats) return;
        
        const files = Object.keys(fileStats);
        
        pdfContainer.innerHTML = `
            <div class="file-list">
                <div class="file-list-header">
                    <h3>📁 Available Documents</h3>
                    <p>Click on any document to view it</p>
                </div>
                <div class="file-grid" id="file-grid">
                    ${files.map((filename, index) => `
                        <div class="file-card ${index === 0 ? 'active' : ''}" 
                             onclick="loadSelectedFile('${filename}')" 
                             data-filename="${filename}">
                            <div class="file-thumbnail" id="thumbnail-${index}">
                                <canvas id="canvas-${index}" width="120" height="160"></canvas>
                                <div class="thumbnail-loading">Loading...</div>
                            </div>
                            <div class="file-card-info">
                                <div class="file-name" title="${filename}">${filename}</div>
                                <div class="file-stats">
                                    <span class="chunk-count">${fileStats[filename].total_chunks} chunks</span>
                                    ${fileStats[filename].available_in_uploads ? 
                                        '<span class="status-available">✅</span>' : 
                                        '<span class="status-unavailable">❌</span>'
                                    }
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        // Generate thumbnails for each file
        files.forEach((filename, index) => {
            if (fileStats[filename].available_in_uploads) {
                generatePdfThumbnail(filename, index);
            } else {
                showThumbnailError(index, "Preview unavailable");
            }
        });
    }
    
    // Load selected file - make this global
    window.loadSelectedFile = function(filename) {
        console.log('Loading selected file:', filename);
        if (documentName) {
            documentName.textContent = filename;
        }
        
        // Clear the file grid and show the main PDF viewer
        const pdfContainer = document.getElementById('pdf-container');
        if (pdfContainer) {
            pdfContainer.innerHTML = `
                <div class="pdf-viewer-controls">
                    <button id="back-to-list" onclick="backToFileList()" class="back-button">
                        ← Back to File List
                    </button>
                </div>
                <canvas id="pdf-render"></canvas>
                <div class="pdf-loading" id="pdf-loading">
                    <div class="spinner"></div>
                    <p>Loading document...</p>
                </div>
            `;
        }
        
        // Re-initialize canvas context
        const canvas = document.getElementById('pdf-render');
        if (canvas) {
            const ctx = canvas.getContext('2d');
        }
        
        const serverUrl = `http://127.0.0.1:5000/uploads/${filename}`;
        console.log('Loading PDF from:', serverUrl);
        
        // Load the PDF for full viewing
        loadPdfFromServer(serverUrl, filename);
        
        // Update active file in grid
        const fileCards = document.querySelectorAll('.file-card');
        fileCards.forEach(card => {
            card.classList.remove('active');
            if (card.dataset.filename === filename) {
                card.classList.add('active');
            }
        });
        
        // Also update active file in old list format (backward compatibility)
        const fileItems = document.querySelectorAll('.file-item');
        fileItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.filename === filename) {
                item.classList.add('active');
            }
        });
    };
    
    // Back to file list function - make this global
    window.backToFileList = function() {
        console.log('Returning to file list');
        
        // Reset document name
        if (documentName) {
            const folderUploadResult = sessionStorage.getItem('folderUploadResult');
            if (folderUploadResult) {
                const folderResult = JSON.parse(folderUploadResult);
                const folderName = folderResult.folder_path ? 
                    folderResult.folder_path.split('/').pop() || folderResult.folder_path.split('\\').pop() : 
                    'Processed Folder';
                documentName.textContent = `📁 ${folderName}`;
            } else {
                documentName.textContent = 'Available Documents';
            }
        }
        
        // Reload available files and show thumbnail grid
        loadAvailableFiles();
    };
    
    // Generate PDF thumbnail
    function generatePdfThumbnail(filename, index) {
        const canvas = document.getElementById(`canvas-${index}`);
        const thumbnailContainer = document.getElementById(`thumbnail-${index}`);
        const loadingIndicator = thumbnailContainer.querySelector('.thumbnail-loading');
        
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const serverUrl = `http://127.0.0.1:5000/uploads/${filename}`;
        
        // Show loading state
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
        
        pdfjsLib.getDocument(serverUrl).promise.then(function(pdf) {
            // Get first page
            return pdf.getPage(1);
        }).then(function(page) {
            // Calculate scale to fit thumbnail
            const viewport = page.getViewport({ scale: 1 });
            const scale = Math.min(120 / viewport.width, 160 / viewport.height);
            const scaledViewport = page.getViewport({ scale });
            
            // Set canvas size
            canvas.height = scaledViewport.height;
            canvas.width = scaledViewport.width;
            
            // Render PDF page into canvas
            const renderContext = {
                canvasContext: ctx,
                viewport: scaledViewport
            };
            
            return page.render(renderContext).promise;
        }).then(function() {
            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            console.log(`Thumbnail generated for ${filename}`);
        }).catch(function(error) {
            console.error(`Error generating thumbnail for ${filename}:`, error);
            showThumbnailError(index, "Failed to load");
        });
    }
    
    // Show thumbnail error
    function showThumbnailError(index, message) {
        const canvas = document.getElementById(`canvas-${index}`);
        const thumbnailContainer = document.getElementById(`thumbnail-${index}`);
        const loadingIndicator = thumbnailContainer.querySelector('.thumbnail-loading');
        
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw error icon
            ctx.fillStyle = '#6c757d';
            ctx.font = '24px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('📄', canvas.width / 2, canvas.height / 2 - 10);
            
            ctx.font = '10px Arial';
            ctx.fillText(message, canvas.width / 2, canvas.height / 2 + 10);
        }
        
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }
    
    // Show no files message
    function showNoFilesMessage() {
        if (documentName) {
            documentName.textContent = 'No documents found';
        }
        
        const pdfContainer = document.getElementById('pdf-container');
        if (pdfContainer) {
            pdfContainer.innerHTML = `
                <div class="no-files">
                    <div class="no-files-icon">📂</div>
                    <h3>No Documents Available</h3>
                    <p>Upload some PDF documents to get started.</p>
                    <button onclick="window.location.href='index.html'" class="upload-btn">
                        Go to Upload Page
                    </button>
                </div>
            `;
        }
        
        // Initialize chat anyway
        initChat();
        
        // Check Ollama connection status
        checkStatus();
    }
    
    // Handle folder upload results
    async function handleFolderUpload(folderResult) {
        console.log('handleFolderUpload called with:', folderResult);
        
        // Get real-time file statistics from the files endpoint
        try {
            const response = await fetch('/files');
            const stats = await response.json();
            
            if (stats.success) {
                // Use real-time stats 
                const actualFileCount = stats.total_files || 0;
                console.log('Real-time file count:', actualFileCount);
                
                // Update document name with actual count
                const folderName = folderResult.folder_path ? folderResult.folder_path.split('/').pop() || folderResult.folder_path.split('\\').pop() : 'Processed Folder';
                if (documentName) {
                    documentName.textContent = `📁 ${folderName} (${actualFileCount} files processed)`;
                }
                
                // Show the file list with thumbnails
                if (stats.file_statistics) {
                    console.log('Showing file list with thumbnails', stats.file_statistics);
                    showFileListInViewer(stats.file_statistics);
                } else {
                    console.log('No file statistics, showing basic summary');
                    showFolderSummary(folderResult);
                }
            }
        } catch (error) {
            console.error('Error fetching file stats:', error);
            // Fall back to basic summary if request fails
            showFolderSummary(folderResult);
        }
        
        console.log('Document name updated to:', documentName ? documentName.textContent : 'null');
        
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
        const loadingIndicator = document.getElementById('pdf-loading');
        const canvas = document.getElementById('pdf-render');
        
        if (!pdfDoc || !canvas) {
            console.error('PDF document or canvas not available');
            return;
        }
        
        if (loadingIndicator) {
            loadingIndicator.style.display = 'flex';
        }
        
        const ctx = canvas.getContext('2d');
        
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
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                if (currentPageEl) {
                    currentPageEl.textContent = pageNumber;
                }
                
                // Update button states
                if (prevButton) {
                    prevButton.disabled = pageNumber <= 1;
                }
                if (nextButton) {
                    nextButton.disabled = pageNumber >= pdfDoc.numPages;
                }
            });
        }).catch(function(error) {
            console.error('Error rendering page:', error);
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
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
        
        // Add predefined question button handlers
        const questionButtons = document.querySelectorAll('.question-btn');
        questionButtons.forEach(button => {
            button.addEventListener('click', function() {
                const question = this.getAttribute('data-question');
                if (question) {
                    queryInput.value = question;
                    sendQuery();
                }
            });
        });
        
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
    if (prevButton) {
        prevButton.addEventListener('click', goPrevPage);
    }
    if (nextButton) {
        nextButton.addEventListener('click', goNextPage);
    }
    if (zoomInButton) {
        zoomInButton.addEventListener('click', zoomIn);
    }
    if (zoomOutButton) {
        zoomOutButton.addEventListener('click', zoomOut);
    }
    
    // Initialize
    init();
});