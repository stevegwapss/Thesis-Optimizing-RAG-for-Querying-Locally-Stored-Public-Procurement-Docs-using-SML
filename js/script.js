// Upload handler script
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const selectFilesBtn = document.getElementById('select-files-btn');
    const uploadBox = document.getElementById('upload-box');
    const fileNameDisplay = document.getElementById('file-name');
    
    // API endpoint
    const UPLOAD_ENDPOINT = '/upload';
    
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
        selectFilesBtn.textContent = 'Uploading...';
        selectFilesBtn.disabled = true;
        
        // Send to server
        fetch(UPLOAD_ENDPOINT, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server returned ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Update pdfData to mark as uploaded
                const pdfData = {
                    uploaded: true,
                    fileName: file.name,
                    url: `/uploads/${file.name}`  // Add URL to the uploaded file
                };
                sessionStorage.setItem('pdfData', JSON.stringify(pdfData));
                
                // Redirect to viewer page
                window.location.href = 'document-viewer.html';
            } else {
                throw new Error(data.message || 'Upload failed');
            }
        })
        .catch(error => {
            showError('Upload failed: ' + error.message);
            selectFilesBtn.textContent = 'Select Files';
            selectFilesBtn.disabled = false;
        });
    }
});