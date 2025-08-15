// DOM Elements
const promptInput = document.getElementById('prompt');
const generateBtn = document.getElementById('generateBtn');
const btnText = document.querySelector('.btn-text');
const loadingSpinner = document.querySelector('.loading-spinner');
const imageResult = document.getElementById('imageResult');
const generatedImage = document.getElementById('generatedImage');
const errorMessage = document.getElementById('errorMessage');
const usedPrompt = document.getElementById('usedPrompt');
const downloadBtn = document.getElementById('downloadBtn');
const shareBtn = document.getElementById('shareBtn');
const recentImages = document.getElementById('recentImages');
const totalImagesCount = document.getElementById('totalImages');
const todayImagesCount = document.getElementById('todayImages');

// API Configuration
const API_BASE_URL = '/api';

// Local Storage Keys
const RECENT_IMAGES_KEY = 'ai_generator_recent_images';
const STATS_KEY = 'ai_generator_stats';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadRecentImages();
    updateStats();
    
    // Event listeners
    generateBtn.addEventListener('click', generateImage);
    downloadBtn.addEventListener('click', downloadImage);
    shareBtn.addEventListener('click', shareImage);
    
    // Enter key support for textarea
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            generateImage();
        }
    });
});

// Generate image function
async function generateImage() {
    const prompt = promptInput.value.trim();
    
    if (!prompt) {
        showError('يرجى إدخال وصف للصورة');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    hideError();
    hideImageResult();
    
    try {
        const response = await fetch(`${API_BASE_URL}/generate-image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt: prompt })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Display the generated image
            displayGeneratedImage(data.image, prompt);
            
            // Save to recent images
            saveToRecentImages(data.image, prompt);
            
            // Update stats
            updateStatsAfterGeneration();
            
            // Clear the input
            promptInput.value = '';
            
        } else {
            // Handle API errors
            let errorMsg = data.error || 'حدث خطأ أثناء توليد الصورة';
            
            if (response.status === 503) {
                errorMsg = 'النموذج قيد التحميل، يرجى المحاولة مرة أخرى خلال دقائق قليلة';
            } else if (response.status === 500 && errorMsg.includes('token')) {
                errorMsg = 'لم يتم تكوين مفتاح Hugging Face API بشكل صحيح';
            }
            
            showError(errorMsg);
        }
        
    } catch (error) {
        console.error('Error generating image:', error);
        showError('حدث خطأ في الاتصال بالخادم. يرجى المحاولة مرة أخرى.');
    } finally {
        setLoadingState(false);
    }
}

// Set loading state
function setLoadingState(isLoading) {
    generateBtn.disabled = isLoading;
    
    if (isLoading) {
        btnText.textContent = 'جاري الإنشاء...';
        loadingSpinner.style.display = 'block';
    } else {
        btnText.textContent = 'إنشاء الصورة';
        loadingSpinner.style.display = 'none';
    }
}

// Display generated image
function displayGeneratedImage(imageData, prompt) {
    generatedImage.src = imageData;
    usedPrompt.textContent = `الوصف المستخدم: "${prompt}"`;
    imageResult.style.display = 'block';
    
    // Scroll to result
    imageResult.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Hide image result
function hideImageResult() {
    imageResult.style.display = 'none';
}

// Download image
function downloadImage() {
    const link = document.createElement('a');
    link.href = generatedImage.src;
    link.download = `ai-generated-image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Share image (copy to clipboard or show share options)
async function shareImage() {
    try {
        // Try to use Web Share API if available
        if (navigator.share) {
            // Convert base64 to blob for sharing
            const response = await fetch(generatedImage.src);
            const blob = await response.blob();
            const file = new File([blob], 'ai-generated-image.png', { type: 'image/png' });
            
            await navigator.share({
                title: 'صورة مولدة بالذكاء الاصطناعي',
                text: usedPrompt.textContent,
                files: [file]
            });
        } else {
            // Fallback: copy image URL to clipboard
            await navigator.clipboard.writeText(window.location.href);
            alert('تم نسخ رابط الصفحة إلى الحافظة');
        }
    } catch (error) {
        console.error('Error sharing:', error);
        alert('حدث خطأ أثناء المشاركة');
    }
}

// Save to recent images
function saveToRecentImages(imageData, prompt) {
    let recentImagesData = getRecentImages();
    
    const newImage = {
        id: Date.now(),
        image: imageData,
        prompt: prompt,
        timestamp: new Date().toISOString()
    };
    
    // Add to beginning of array
    recentImagesData.unshift(newImage);
    
    // Keep only last 12 images
    recentImagesData = recentImagesData.slice(0, 12);
    
    // Save to localStorage
    localStorage.setItem(RECENT_IMAGES_KEY, JSON.stringify(recentImagesData));
    
    // Update display
    loadRecentImages();
}

// Get recent images from localStorage
function getRecentImages() {
    try {
        const data = localStorage.getItem(RECENT_IMAGES_KEY);
        return data ? JSON.parse(data) : [];
    } catch (error) {
        console.error('Error loading recent images:', error);
        return [];
    }
}

// Load and display recent images
function loadRecentImages() {
    const recentImagesData = getRecentImages();
    recentImages.innerHTML = '';
    
    if (recentImagesData.length === 0) {
        recentImages.innerHTML = '<p style="text-align: center; color: #666; padding: 2rem;">لا توجد صور مولدة بعد</p>';
        return;
    }
    
    recentImagesData.forEach(imageData => {
        const imageElement = document.createElement('div');
        imageElement.className = 'recent-image';
        imageElement.innerHTML = `
            <img src="${imageData.image}" alt="${imageData.prompt}" title="${imageData.prompt}">
        `;
        
        // Click to view full image
        imageElement.addEventListener('click', () => {
            displayGeneratedImage(imageData.image, imageData.prompt);
        });
        
        recentImages.appendChild(imageElement);
    });
}

// Get stats from localStorage
function getStats() {
    try {
        const data = localStorage.getItem(STATS_KEY);
        return data ? JSON.parse(data) : { total: 0, today: 0, lastDate: null };
    } catch (error) {
        console.error('Error loading stats:', error);
        return { total: 0, today: 0, lastDate: null };
    }
}

// Update stats after generation
function updateStatsAfterGeneration() {
    let stats = getStats();
    const today = new Date().toDateString();
    
    // Reset today count if it's a new day
    if (stats.lastDate !== today) {
        stats.today = 0;
        stats.lastDate = today;
    }
    
    stats.total += 1;
    stats.today += 1;
    
    // Save stats
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
    
    // Update display
    updateStats();
}

// Update stats display
function updateStats() {
    const stats = getStats();
    const today = new Date().toDateString();
    
    // Reset today count if it's a new day
    if (stats.lastDate !== today) {
        stats.today = 0;
    }
    
    totalImagesCount.textContent = stats.total;
    todayImagesCount.textContent = stats.today;
}

// Utility function to format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Error handling for images
document.addEventListener('error', function(e) {
    if (e.target.tagName === 'IMG') {
        e.target.style.display = 'none';
        console.error('Image failed to load:', e.target.src);
    }
}, true);

// Add some helpful keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to generate
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (document.activeElement === promptInput) {
            generateImage();
        }
    }
    
    // Escape to hide error
    if (e.key === 'Escape') {
        hideError();
    }
});

