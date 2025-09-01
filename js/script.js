// Upload handler script
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const selectFilesBtn = document.getElementById('select-files-btn');
    const uploadBox = document.getElementById('upload-box');
    const fileNameDisplay = document.getElementById('file-name');
    
    // API endpoint
    const UPLOAD_ENDPOINT = 'http://127.0.0.1:5000/upload';
    
    // Setup event listeners
    selectFilesBtn.addEventListener('click', () => fileInput.click());
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
    
    // Process selected file
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
    
    // Show error message
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        uploadBox.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 3000);
    }
    
    // Upload file to server and redirect to viewer page
    function uploadFile(file) {
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading state
        selectFilesBtn.textContent = 'Processing...';
        selectFilesBtn.disabled = true;
        
        console.log('Starting upload for file:', file.name);
        
        // Send to server
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
                // Store file info in session storage
                sessionStorage.setItem('uploadedPdf', file.name);
                const pdfData = {
                    uploaded: true,
                    fileName: file.name,
                    url: `http://127.0.0.1:5000/uploads/${file.name}`,
                    chunks_added: data.chunks_added
                };
                sessionStorage.setItem('pdfData', JSON.stringify(pdfData));
                
                // Show success
                selectFilesBtn.textContent = 'Success! Redirecting...';
                selectFilesBtn.style.backgroundColor = '#4CAF50';
                
                console.log('Upload successful, redirecting to document-viewer.html');
                
                // Immediate redirect
                window.location.href = 'document-viewer.html';
                
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        })
        .catch(error => {
            console.error('Upload failed:', error);
            showError('Upload failed: ' + error.message);
            selectFilesBtn.textContent = 'Select Files';
            selectFilesBtn.disabled = false;
            selectFilesBtn.style.backgroundColor = '';
        });
    }
});