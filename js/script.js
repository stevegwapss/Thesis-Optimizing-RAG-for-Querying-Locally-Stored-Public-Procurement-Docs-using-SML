// Enhanced upload handler script with folder support
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const selectFilesBtn = document.getElementById('select-files-btn');
    const selectFolderBtn = document.getElementById('select-folder-btn');
    const folderPathInput = document.getElementById('folder-path-input');
    const uploadBox = document.getElementById('upload-box');
    const fileNameDisplay = document.getElementById('file-name');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    // API endpoints
    const UPLOAD_ENDPOINT = 'http://127.0.0.1:5000/upload';
    const FOLDER_ENDPOINT = 'http://127.0.0.1:5000/upload-folder';
    const PROGRESS_ENDPOINT = 'http://127.0.0.1:5000/upload-progress';
    
    // Setup event listeners
    selectFilesBtn.addEventListener('click', () => fileInput.click());
    if (selectFolderBtn) {
        selectFolderBtn.addEventListener('click', handleFolderSelection);
    }
    fileInput.addEventListener('change', handleFileSelect);
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);
    
    // Handle drag over event
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.add('drag-over');
    }
    
    // Handle drag leave event
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');
    }
    
    // Handle drop event
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadBox.classList.remove('drag-over');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files[0]);
        }
    }
    
    // Handle file selection
    function handleFileSelect(e) {
        if (e.target.files && e.target.files.length > 0) {
            handleFiles(e.target.files[0]);
        }
    }
    
    // Enhanced folder selection handler with better UX
    function handleFolderSelection() {
        console.log('Folder selection button clicked!'); // Debug
        
        // Create a simple, reliable prompt for folder path
        const folderPath = prompt(
            'Enter the full path to your folder containing PDF files:\n\n' +
            'Examples:\n' +
            'Windows: C:/Users/YourName/Documents/PDFs\n' +
            'Or: C:\\\\Users\\\\YourName\\\\Documents\\\\PDFs\n\n' +
            'Make sure to use the complete path to your folder:'
        );
        
        if (folderPath && folderPath.trim()) {
            console.log('User entered folder path:', folderPath.trim());
            uploadFolder(folderPath.trim());
        } else {
            console.log('User cancelled or entered empty path');
        }
    }

    // Enhanced folder upload function with better error handling
    function uploadFolder(folderPath) {
        console.log('Starting uploadFolder function with path:', folderPath);
        
        // Normalize path separators
        folderPath = folderPath.replace(/\\/g, '/');
        
        // Show folder name and start processing
        fileNameDisplay.textContent = `Processing folder: ${folderPath}`;
        fileNameDisplay.style.display = 'block';
        
        // Show progress container
        showProgress();
        
        // Disable buttons
        selectFilesBtn.textContent = 'Processing Folder...';
        selectFilesBtn.disabled = true;
        if (selectFolderBtn) {
            selectFolderBtn.disabled = true;
        }
        
        console.log('Sending request to:', FOLDER_ENDPOINT);
        console.log('Request payload:', { 
            folder_path: folderPath,
            max_workers: 3
        });
        
        // Send folder path to server
        fetch(FOLDER_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                folder_path: folderPath,
                max_workers: 3  // Limit concurrent processing
            })
        })
        .then(response => {
            console.log('Folder upload response status:', response.status);
            console.log('Response headers:', response.headers);
            
            if (!response.ok) {
                return response.text().then(text => {
                    console.error('Error response text:', text);
                    try {
                        const errorData = JSON.parse(text);
                        throw new Error(errorData.error || `Server error: ${response.status}`);
                    } catch (parseError) {
                        throw new Error(`Server error: ${response.status} - ${text}`);
                    }
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Folder upload response data:', data);
            
            if (data.success) {
                console.log('Starting progress monitoring...');
                // Start monitoring progress
                monitorProgress();
            } else {
                throw new Error(data.message || 'Folder upload failed');
            }
        })
        .catch(error => {
            console.error('Folder upload failed:', error);
            showError('Folder upload failed: ' + error.message);
            resetUploadButtons();
            hideProgress();
        });
    }
    
    // Handle file processing
    function handleFiles(file) {
        // Check if file is PDF
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showError("Please select a PDF file");
            return;
        }
        
        // Check file size (limit to 25MB)
        if (file.size > 25 * 1024 * 1024) {
            showError("File size exceeds 25MB limit");
            return;
        }
        
        // Display file name
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.style.display = 'block';
        
        // Read file as base64 and store in sessionStorage
        const reader = new FileReader();
        reader.onload = function(e) {
            const base64Data = e.target.result;
            
            // Store file metadata in sessionStorage
            sessionStorage.setItem('uploadedPdf', file.name);
            sessionStorage.setItem('pdfLastModified', file.lastModified);
            
            // Store the PDF data for document-viewer.js to use
            const pdfData = {
                fileName: file.name,
                base64: base64Data,
                uploaded: false
            };
            sessionStorage.setItem('pdfData', JSON.stringify(pdfData));
            
            // Upload file to server
            uploadFile(file);
        };
        reader.readAsDataURL(file);
    }
    
    // Progress monitoring
    function monitorProgress() {
        const progressInterval = setInterval(() => {
            fetch(PROGRESS_ENDPOINT)
                .then(response => response.json())
                .then(progress => {
                    updateProgress(progress);
                    
                    if (progress.status === 'completed') {
                        clearInterval(progressInterval);
                        handleProcessingComplete(progress);
                    } else if (progress.status === 'error' || progress.status === 'cancelled') {
                        clearInterval(progressInterval);
                        handleProcessingError(progress);
                    }
                })
                .catch(error => {
                    console.error('Progress check failed:', error);
                    clearInterval(progressInterval);
                    handleProcessingError({ errors: [error.message] });
                });
        }, 1000); // Check progress every second
    }
    
    // Update progress display
    function updateProgress(progress) {
        if (progress.total_files > 0) {
            const percentage = (progress.processed_files / progress.total_files) * 100;
            progressBar.style.width = percentage + '%';
            
            let statusText = `Processing ${progress.processed_files}/${progress.total_files} files`;
            if (progress.current_file) {
                statusText += ` - Current: ${progress.current_file}`;
            }
            if (progress.eta_seconds) {
                const eta = Math.round(progress.eta_seconds);
                statusText += ` - ETA: ${eta}s`;
            }
            
            progressText.textContent = statusText;
        }
    }
    
    // Handle processing completion
    function handleProcessingComplete(progress) {
        hideProgress();
        
        selectFilesBtn.textContent = 'Success! Redirecting...';
        selectFilesBtn.style.backgroundColor = '#4CAF50';
        
        // Store successful upload info
        sessionStorage.setItem('uploadedFolder', 'true');
        sessionStorage.setItem('folderUploadResult', JSON.stringify(progress));
        
        console.log('Folder processing completed, redirecting to document-viewer.html');
        
        setTimeout(() => {
            window.location.href = 'document-viewer.html';
        }, 1500);
    }
    
    // Handle processing errors
    function handleProcessingError(progress) {
        hideProgress();
        
        let errorMessage = 'Processing failed';
        if (progress.errors && progress.errors.length > 0) {
            errorMessage += ': ' + progress.errors.join(', ');
        }
        
        showError(errorMessage);
        resetUploadButtons();
    }
    
    // Utility functions
    function showProgress() {
        if (progressContainer) {
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            progressText.textContent = 'Initializing...';
        }
    }
    
    function hideProgress() {
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    }
    
    function resetUploadButtons() {
        selectFilesBtn.textContent = 'Select Files';
        selectFilesBtn.disabled = false;
        selectFilesBtn.style.backgroundColor = '';
        
        if (selectFolderBtn) {
            selectFolderBtn.disabled = false;
        }
    }
    
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            border: 1px solid #f5c6cb;
            font-size: 14px;
            line-height: 1.4;
        `;
        errorDiv.textContent = message;
        
        uploadBox.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 8000);
    }
    
    // Enhanced single file upload (existing function with progress support)
    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        selectFilesBtn.textContent = 'Processing...';
        selectFilesBtn.disabled = true;
        
        console.log('Starting upload for file:', file.name);
        
        fetch(UPLOAD_ENDPOINT, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Response received:', response.status, response.ok);
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload response:', data);
            
            if (data.success) {
                sessionStorage.setItem('uploadedPdf', file.name);
                const pdfData = {
                    uploaded: true,
                    fileName: file.name,
                    url: `http://127.0.0.1:5000/uploads/${file.name}`,
                    chunks_added: data.chunks_added
                };
                sessionStorage.setItem('pdfData', JSON.stringify(pdfData));
                
                selectFilesBtn.textContent = 'Success! Redirecting...';
                selectFilesBtn.style.backgroundColor = '#4CAF50';
                
                console.log('Upload successful, redirecting to document-viewer.html');
                window.location.href = 'document-viewer.html';
                
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        })
        .catch(error => {
            console.error('Upload failed:', error);
            showError('Upload failed: ' + error.message);
            resetUploadButtons();
        });
    }
});