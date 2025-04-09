/**
 * WESDEABEN PDF Summarizer JavaScript
 * Adds interactivity to the PDF summary page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Toggle suggested questions section
    const collapseButton = document.querySelector('.collapse-button');
    const questionsList = document.querySelector('.questions-list');
    
    collapseButton.addEventListener('click', function() {
        questionsList.style.display = questionsList.style.display === 'none' ? 'block' : 'none';
        
        // Toggle icon
        const icon = collapseButton.querySelector('i');
        if (icon.classList.contains('fa-chevron-up')) {
            icon.classList.remove('fa-chevron-up');
            icon.classList.add('fa-chevron-down');
        } else {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-up');
        }
    });
    
    // Handle question clicks
    const questionButtons = document.querySelectorAll('.question-button');
    const chatInput = document.querySelector('.chat-input input');
    
    questionButtons.forEach(button => {
        button.addEventListener('click', function() {
            const questionText = this.parentElement.querySelector('.question-text').textContent;
            chatInput.value = questionText;
            chatInput.focus();
        });
    });
    
    // Handle send button click
    const sendButton = document.querySelector('.send-button');
    
    sendButton.addEventListener('click', function() {
        const message = chatInput.value.trim();
        if (message) {
            // In a real application, this would send the message to a backend
            console.log('Sending message:', message);
            
            // Clear input after sending
            chatInput.value = '';
            chatInput.focus();
        }
    });
    
    // Handle Enter key in chat input
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });
    
    // PDF navigation controls
    const prevButton = document.querySelector('.pdf-navigation .nav-button:first-child');
    const nextButton = document.querySelector('.pdf-navigation .nav-button:last-child');
    const pageNumberInput = document.querySelector('.page-number');
    const totalPages = 33; // From the screenshot
    
    let currentPage = 1;
    
    prevButton.addEventListener('click', function() {
        if (currentPage > 1) {
            currentPage--;
            updatePageDisplay();
        }
    });
    
    nextButton.addEventListener('click', function() {
        if (currentPage < totalPages) {
            currentPage++;
            updatePageDisplay();
        }
    });
    
    pageNumberInput.addEventListener('change', function() {
        const newPage = parseInt(this.value);
        if (!isNaN(newPage) && newPage >= 1 && newPage <= totalPages) {
            currentPage = newPage;
        }
        updatePageDisplay();
    });
    
    function updatePageDisplay() {
        pageNumberInput.value = currentPage;
        
        // In a real application, this would load the corresponding page
        console.log('Current page:', currentPage);
    }
    
    // PDF zoom controls
    const zoomOutButton = document.querySelector('.pdf-tools .tool-button:nth-child(1)');
    const zoomInButton = document.querySelector('.pdf-tools .tool-button:nth-child(2)');
    const fullscreenButton = document.querySelector('.pdf-tools .tool-button:nth-child(3)');
    
    let zoomLevel = 100; // percentage
    
    zoomOutButton.addEventListener('click', function() {
        if (zoomLevel > 50) {
            zoomLevel -= 10;
            updateZoom();
        }
    });
    
    zoomInButton.addEventListener('click', function() {
        if (zoomLevel < 200) {
            zoomLevel += 10;
            updateZoom();
        }
    });
    
    function updateZoom() {
        // In a real application, this would adjust the PDF zoom
        console.log('Zoom level:', zoomLevel + '%');
    }
    
    fullscreenButton.addEventListener('click', function() {
        // In a real application, this would toggle fullscreen mode
        console.log('Toggle fullscreen');
    });
    
    // Download button
    const downloadButton = document.querySelector('.download-button');
    
    downloadButton.addEventListener('click', function() {
        // In a real application, this would trigger the PDF download
        console.log('Downloading PDF');
    });
    
    // Feedback buttons
    const feedbackButtons = document.querySelectorAll('.feedback-button');
    
    feedbackButtons.forEach(button => {
        button.addEventListener('click', function() {
            // In a real application, this would send feedback
            const isPositive = this.querySelector('i').classList.contains('fa-smile');
            console.log('Feedback:', isPositive ? 'positive' : 'negative');
            
            // Visual feedback
            this.style.color = '#c1a06d';
            setTimeout(() => {
                this.style.color = '';
            }, 1000);
        });
    });
    
    // Copy button
    const copyButton = document.querySelector('.copy-button');
    
    copyButton.addEventListener('click', function() {
        // In a real application, this would copy the summary to clipboard
        console.log('Copying summary to clipboard');
        
        // Visual feedback
        this.style.color = '#c1a06d';
        setTimeout(() => {
            this.style.color = '';
        }, 1000);
    });
});
