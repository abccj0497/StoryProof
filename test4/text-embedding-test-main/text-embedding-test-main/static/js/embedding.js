// Embedding Similarity Analyzer - JavaScript Module

const modelCheckboxesContainer = document.getElementById('modelCheckboxes');
const sentenceInputsContainer = document.getElementById('sentenceInputs');
const addSentenceBtn = document.getElementById('addSentenceBtn');
const executeBtn = document.getElementById('executeBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const errorToast = document.getElementById('errorToast');
const errorMessage = document.getElementById('errorMessage');

let models = [];
let sentenceCount = 0;

// Initialize
async function initialize() {
    await loadModels();
    addSentenceInput();
    addSentenceInput();
    setupEventListeners();
}

// Load model list
async function loadModels() {
    try {
        const response = await fetch('/api/embedding/models');
        const data = await response.json();
        models = data.models;
        renderModelCheckboxes();
    } catch (error) {
        showError('Failed to load model list.');
        console.error('Error loading models:', error);
    }
}

// Render model checkboxes
function renderModelCheckboxes() {
    modelCheckboxesContainer.innerHTML = models.map((model, index) => `
        <label class="model-checkbox selected" data-model="${model.name}">
            <input type="checkbox" name="model" value="${model.name}" checked>
            <span class="checkbox-custom">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#0d0d0d" stroke-width="3">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            </span>
            <span class="model-info">
                <span class="model-name">${model.name}</span>
                <span class="model-dimension">Dimension: ${model.dimension}</span>
            </span>
        </label>
    `).join('');

    // Add checkbox event listeners
    document.querySelectorAll('.model-checkbox').forEach(checkbox => {
        checkbox.addEventListener('click', (e) => {
            const input = checkbox.querySelector('input');
            input.checked = !input.checked;
            checkbox.classList.toggle('selected', input.checked);
        });
    });
}

// Add sentence input field
function addSentenceInput() {
    sentenceCount++;
    const wrapper = document.createElement('div');
    wrapper.className = 'sentence-input-wrapper';
    wrapper.dataset.index = sentenceCount;
    
    wrapper.innerHTML = `
        <span class="sentence-label">Sentence ${sentenceCount}</span>
        <input 
            type="text" 
            class="sentence-input" 
            placeholder="Enter a sentence to compare..."
            data-index="${sentenceCount}"
        >
        <button type="button" class="btn-remove" title="Remove">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
        </button>
    `;
    
    sentenceInputsContainer.appendChild(wrapper);
    
    // Remove button event
    wrapper.querySelector('.btn-remove').addEventListener('click', () => {
        removeSentenceInput(wrapper);
    });
    
    // Focus on newly added input field
    wrapper.querySelector('.sentence-input').focus();
}

// Remove sentence input field
function removeSentenceInput(wrapper) {
    const inputs = sentenceInputsContainer.querySelectorAll('.sentence-input-wrapper');
    
    if (inputs.length <= 2) {
        showError('At least 2 sentences are required.');
        return;
    }
    
    wrapper.remove();
    updateSentenceLabels();
}

// Update sentence labels

function updateSentenceLabels() {
    const wrappers = sentenceInputsContainer.querySelectorAll('.sentence-input-wrapper');
    wrappers.forEach((wrapper, index) => {
        const label = wrapper.querySelector('.sentence-label');
        const input = wrapper.querySelector('.sentence-input');
        label.textContent = `Sentence ${index + 1}`;
        input.dataset.index = index + 1;
    });
    sentenceCount = wrappers.length;
}

// Setup event listeners
function setupEventListeners() {
    addSentenceBtn.addEventListener('click', addSentenceInput);
    executeBtn.addEventListener('click', executeComparison);
}

// Execute similarity comparison
async function executeComparison() {
    // Collect selected models
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox.selected input'))
        .map(input => input.value);
    
    // Collect input sentences
    const sentences = Array.from(document.querySelectorAll('.sentence-input'))
        .map(input => input.value.trim());
    
    // Validation
    if (selectedModels.length === 0) {
        showError('Please select at least one model.');
        return;
    }
    
    if (sentences.length < 2) {
        showError('At least 2 sentences are required.');
        return;
    }
    
    const emptyIndex = sentences.findIndex(s => !s);
    if (emptyIndex !== -1) {
        showError(`Sentence ${emptyIndex + 1} is empty.`);
        document.querySelectorAll('.sentence-input')[emptyIndex].focus();
        return;
    }
    
    // Loading state
    setLoading(true);
    
    try {
        const response = await fetch('/api/embedding/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sentences: sentences,
                models: selectedModels
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'An error occurred during analysis.');
        }
        
        const data = await response.json();
        renderResults(data);
        
    } catch (error) {
        showError(error.message);
        console.error('Error:', error);
    } finally {
        setLoading(false);
    }
}

// Render results
function renderResults(data) {
    resultsSection.style.display = 'block';
    
    resultsContainer.innerHTML = data.results.map(result => `
        <div class="result-card">
            <div class="result-header">
                <span class="result-model-name">${result.model_name}</span>
                <span class="result-dimension">Dimension: ${result.dimension}</span>
            </div>
            
            <div class="similarity-table-wrapper">
                <h3 style="font-size: 0.95rem; color: var(--color-text-primary); margin-bottom: 0.75rem; font-weight: 500;">Cosine Similarity</h3>
                ${renderSimilarityTable(result.cosine_similarity, data.sentences, 'cosine')}
            </div>
            
            <div class="similarity-table-wrapper" style="margin-top: 1.5rem;">
                <h3 style="font-size: 0.95rem; color: var(--color-text-primary); margin-bottom: 0.75rem; font-weight: 500;">Euclidean Distance</h3>
                ${renderSimilarityTable(result.normalized_euclidean_distance, data.sentences, 'normalized_euclidean')}
            </div>
        </div>
    `).join('');
    
    // Scroll to results section
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Render similarity table
function renderSimilarityTable(matrix, sentences, metricType) {
    const truncate = (str, maxLen = 30) => {
        return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
    };
    
    const getSimilarityClass = (value, i, j, metricType) => {
        if (i === j) return 'diagonal';
        
        if (metricType === 'cosine') {
            // Cosine similarity: higher is more similar (0~1)
            if (value >= 0.8) return 'high-similarity';
            if (value >= 0.5) return 'medium-similarity';
            return 'low-similarity';
        } else {
            // Euclidean distance: lower is more similar (normalized range 0~2)
            if (value <= 0.5) return 'high-similarity';
            if (value <= 1.0) return 'medium-similarity';
            return 'low-similarity';
        }
    };
    
    let html = '<table class="similarity-table">';
    
    // Header
    html += '<thead><tr><th></th>';
    sentences.forEach((s, i) => {
        html += `<th class="sentence-header" title="${escapeHtml(s)}">Sentence ${i + 1}</th>`;
    });
    html += '</tr></thead>';
    
    // Body
    html += '<tbody>';
    matrix.forEach((row, i) => {
        html += `<tr><th class="sentence-header" title="${escapeHtml(sentences[i])}">Sentence ${i + 1}</th>`;
        row.forEach((value, j) => {
            const similarityClass = getSimilarityClass(value, i, j, metricType);
            html += `<td class="${similarityClass}">${value.toFixed(4)}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    
    html += '</table>';
    
    // Add sentence list only to the first table
    if (metricType === 'cosine') {
        html += '<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--color-border);">';
        html += '<div style="font-size: 0.8rem; color: var(--color-text-muted); margin-bottom: 0.5rem;">Input Sentences:</div>';
        sentences.forEach((s, i) => {
            html += `<div style="font-size: 0.85rem; color: var(--color-text-secondary); margin-bottom: 0.25rem;">
                <span style="color: var(--color-accent);">Sentence ${i + 1}:</span> ${escapeHtml(s)}
            </div>`;
        });
        html += '</div>';
    }
    
    return html;
}

// Set loading state
function setLoading(loading) {
    executeBtn.disabled = loading;
    const btnText = executeBtn.querySelector('.btn-text');
    const btnLoading = executeBtn.querySelector('.btn-loading');
    
    if (loading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
    } else {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

// Show error toast
function showError(message) {
    errorMessage.textContent = message;
    errorToast.style.display = 'block';
    
    setTimeout(() => {
        errorToast.style.display = 'none';
    }, 4000);
}

// HTML escape
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', initialize);

