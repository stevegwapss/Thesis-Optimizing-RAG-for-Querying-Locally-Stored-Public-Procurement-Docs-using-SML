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
        UPLOAD: '/upload',
        QUERY: '/query',
        STATUS: '/status'
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
            pdfSource = `/uploads/${source}`;
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
        
        // Add welcome message
        //addMessage('system', 'Welcome! You can ask questions about the document content.');
    }
    
    // Send query to AI
    function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Add user message to chat
        addMessage('user', query);
        
        // Clear input
        queryInput.value = '';
        
        // Show thinking indicator
        showThinking();
        
        // Send query to backend
        fetch(API_ENDPOINTS.QUERY, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // Remove thinking message
            removeThinking();
            
            if (data.error) {
                addMessage('system', `Error: ${data.error}`);
            } else {
                addMessage('assistant', data.response);
            }
        })
        .catch(error => {
            removeThinking();
            console.error('Query error:', error);
            addMessage('system', 'Failed to get a response. Please try again.');
        });
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