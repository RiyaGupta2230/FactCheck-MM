// Global variables
let isLoading = false;
let comicSounds = {
    pow: '💥',
    zap: '⚡',
    boom: '💣',
    star: '⭐'
};

// Initialize application with comic flair
document.addEventListener('DOMContentLoaded', function() {
    addComicEntrance();
    checkSystemHealth();
    setupEventListeners();
    addComicInteractions();
});

function addComicEntrance() {
    document.body.classList.add('comic-entrance');
    setTimeout(() => {
        document.body.classList.remove('comic-entrance');
    }, 1000);
}

function addComicInteractions() {
    // Add comic sound effects to buttons
    document.querySelectorAll('.btn').forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            btn.style.transform = 'translate(-2px, -2px) scale(1.05)';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.transform = 'translate(0, 0) scale(1)';
        });
    });

    // Add typing animation to textareas
    document.querySelectorAll('textarea').forEach(textarea => {
        textarea.addEventListener('focus', () => {
            textarea.parentElement.classList.add('comic-focus');
        });
        textarea.addEventListener('blur', () => {
            textarea.parentElement.classList.remove('comic-focus');
        });
    });
}

function setupEventListeners() {
    // Enhanced enter key listeners with comic feedback
    document.getElementById('sarcasmText').addEventListener('keypress', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            showComicHint('💥 CTRL+ENTER activated!');
            detectSarcasm();
        }
    });
    
    document.getElementById('sentence1').addEventListener('keypress', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            showComicHint('⚡ ZAP! Quick analysis!');
            checkParaphrase();
        }
    });
    
    document.getElementById('sentence2').addEventListener('keypress', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            showComicHint('💫 BOOM! Processing!');
            checkParaphrase();
        }
    });
    
    document.getElementById('claimText').addEventListener('keypress', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            showComicHint('🎯 KAPOW! Fact checking!');
            checkFact();
        }
    });
}

function showComicHint(message) {
    const hint = document.createElement('div');
    hint.className = 'comic-hint';
    hint.textContent = message;
    document.body.appendChild(hint);
    
    setTimeout(() => {
        hint.remove();
    }, 2000);
}

// Enhanced result display functions with comic styling
function displaySarcasmResult(result) {
    const resultDiv = document.getElementById('sarcasmResult');
    const comicVerdict = result.is_sarcastic ? 
        { emoji: '😏', sound: '💥', verdict: 'SARCASTIC!', color: 'warning' } :
        { emoji: '😊', sound: '⭐', verdict: 'SINCERE!', color: 'success' };
    
    const confidencePercent = Math.round(result.confidence * 100);
    const sarcasticPercent = Math.round(result.probabilities.sarcastic * 100);
    const nonSarcasticPercent = Math.round(result.probabilities.non_sarcastic * 100);
    
    resultDiv.innerHTML = `
        <div class="alert comic-result fade-in">
            <div class="comic-result-header">
                <div class="comic-verdict ${comicVerdict.color}">
                    <span class="comic-emoji">${comicVerdict.emoji}</span>
                    <span class="comic-sound">${comicVerdict.sound}</span>
                    <span class="comic-verdict-text">${comicVerdict.verdict}</span>
                </div>
                <div class="comic-confidence">
                    🎯 Confidence: ${confidencePercent}%
                </div>
            </div>
            
            <div class="comic-probability-section">
                <div class="comic-probability-item">
                    <div class="comic-probability-label">
                        😏 Sarcasm Level: ${sarcasticPercent}%
                    </div>
                    <div class="probability-bar comic-bar">
                        <div class="probability-fill bg-warning comic-fill" style="width: ${sarcasticPercent}%">
                            ${sarcasticPercent > 20 ? sarcasticPercent + '%' : ''}
                        </div>
                    </div>
                </div>
                
                <div class="comic-probability-item">
                    <div class="comic-probability-label">
                        😊 Sincerity Level: ${nonSarcasticPercent}%
                    </div>
                    <div class="probability-bar comic-bar">
                        <div class="probability-fill bg-success comic-fill" style="width: ${nonSarcasticPercent}%">
                            ${nonSarcasticPercent > 20 ? nonSarcasticPercent + '%' : ''}
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="comic-explanation">
                <i class="fas fa-lightbulb"></i>
                <span class="comic-explanation-text">${result.explanation}</span>
            </div>
        </div>
    `;
}

function displayParaphraseResult(result) {
    const resultDiv = document.getElementById('paraphraseResult');
    
    if (result.status) {
        resultDiv.innerHTML = `
            <div class="alert comic-result comic-info fade-in">
                <div class="comic-info-header">
                    <i class="fas fa-rocket fa-2x"></i>
                    <div class="comic-info-text">
                        <div class="comic-title">🚀 MILESTONE 2 LOADING...</div>
                        <div class="comic-subtitle">${result.status}</div>
                    </div>
                </div>
            </div>
        `;
        return;
    }
    
    const comicVerdict = result.is_paraphrase ? 
        { emoji: '✅', sound: '💥', verdict: 'PARAPHRASE!', color: 'success' } :
        { emoji: '❌', sound: '💫', verdict: 'DIFFERENT!', color: 'danger' };
    
    resultDiv.innerHTML = `
        <div class="alert comic-result fade-in">
            <div class="comic-result-header">
                <div class="comic-verdict ${comicVerdict.color}">
                    <span class="comic-emoji">${comicVerdict.emoji}</span>
                    <span class="comic-sound">${comicVerdict.sound}</span>
                    <span class="comic-verdict-text">${comicVerdict.verdict}</span>
                </div>
            </div>
            
            <div class="comic-stats">
                <div class="comic-stat-item">
                    <div class="comic-stat-label">🎯 Similarity Score</div>
                    <div class="comic-stat-value">${Math.round(result.similarity_score * 100)}%</div>
                </div>
                <div class="comic-stat-item">
                    <div class="comic-stat-label">⚡ Confidence</div>
                    <div class="comic-stat-value">${Math.round(result.confidence * 100)}%</div>
                </div>
            </div>
        </div>
    `;
}

function displayFactCheckResult(result) {
    const resultDiv = document.getElementById('factcheckResult');
    
    if (result.status) {
        resultDiv.innerHTML = `
            <div class="alert comic-result comic-info fade-in">
                <div class="comic-info-header">
                    <i class="fas fa-shield-alt fa-2x"></i>
                    <div class="comic-info-text">
                        <div class="comic-title">🛡️ MILESTONE 3 PREPARING...</div>
                        <div class="comic-subtitle">${result.status}</div>
                    </div>
                </div>
            </div>
        `;
        return;
    }
    
    let comicVerdict = { emoji: '❓', sound: '💭', verdict: 'UNKNOWN', color: 'secondary' };
    
    if (result.verdict === 'SUPPORTS') {
        comicVerdict = { emoji: '✅', sound: '💥', verdict: 'TRUE!', color: 'success' };
    } else if (result.verdict === 'REFUTES') {
        comicVerdict = { emoji: '❌', sound: '💣', verdict: 'FALSE!', color: 'danger' };
    } else if (result.verdict === 'NOT_ENOUGH_INFO') {
        comicVerdict = { emoji: '⚠️', sound: '💫', verdict: 'UNCLEAR!', color: 'warning' };
    }
    
    let evidenceHtml = '';
    if (result.evidence && result.evidence.length > 0) {
        evidenceHtml = `
            <div class="comic-evidence">
                <div class="comic-evidence-title">
                    <i class="fas fa-scroll"></i> Evidence Found:
                </div>
                <ul class="comic-evidence-list">
                    ${result.evidence.map(ev => `<li class="comic-evidence-item">${ev}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    resultDiv.innerHTML = `
        <div class="alert comic-result fade-in">
            <div class="comic-result-header">
                <div class="comic-verdict ${comicVerdict.color}">
                    <span class="comic-emoji">${comicVerdict.emoji}</span>
                    <span class="comic-sound">${comicVerdict.sound}</span>
                    <span class="comic-verdict-text">${comicVerdict.verdict}</span>
                </div>
                <div class="comic-confidence">
                    🎯 Confidence: ${Math.round(result.confidence * 100)}%
                </div>
            </div>
            ${evidenceHtml}
        </div>
    `;
}

function updateSystemStatus(data) {
    const statusDiv = document.getElementById('systemStatus');
    const models = data.models_loaded || {};
    
    const getStatusEmoji = (status) => status ? '🟢' : '🔴';
    const getStatusText = (status) => status ? 'READY' : 'LOADING';
    const getStatusClass = (status) => status ? 'bg-success' : 'bg-secondary';
    
    let statusHtml = `
        <div class="row comic-status-grid">
            <div class="col-md-3">
                <div class="comic-status-item">
                    <div class="comic-status-icon">${data.status === 'healthy' ? '💚' : '💔'}</div>
                    <div class="comic-status-label">SYSTEM</div>
                    <div class="badge ${data.status === 'healthy' ? 'bg-success' : 'bg-danger'} comic-badge">
                        ${data.status === 'healthy' ? 'ONLINE' : 'OFFLINE'}
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="comic-status-item">
                    <div class="comic-status-icon">${getStatusEmoji(models.sarcasm)}</div>
                    <div class="comic-status-label">SARCASM</div>
                    <div class="badge ${getStatusClass(models.sarcasm)} comic-badge">
                        ${getStatusText(models.sarcasm)}
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="comic-status-item">
                    <div class="comic-status-icon">${getStatusEmoji(models.paraphrase)}</div>
                    <div class="comic-status-label">PARAPHRASE</div>
                    <div class="badge ${getStatusClass(models.paraphrase)} comic-badge">
                        ${getStatusText(models.paraphrase)}
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="comic-status-item">
                    <div class="comic-status-icon">${getStatusEmoji(models.fact_check)}</div>
                    <div class="comic-status-label">FACT CHECK</div>
                    <div class="badge ${getStatusClass(models.fact_check)} comic-badge">
                        ${getStatusText(models.fact_check)}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    statusDiv.innerHTML = statusHtml;
}

// Enhanced loading function with comic effects
function setLoading(button, loading) {
    isLoading = loading;
    if (loading) {
        button.disabled = true;
        button.classList.add('loading', 'comic-loading');
        
        const loadingMessages = ['💥 PROCESSING...', '⚡ ANALYZING...', '🚀 COMPUTING...', '💫 WORKING...'];
        const randomMessage = loadingMessages[Math.floor(Math.random() * loadingMessages.length)];
        
        button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${randomMessage}`;
    } else {
        button.disabled = false;
        button.classList.remove('loading', 'comic-loading');
        
        // Restore original text with comic flair
        if (button.id === 'sarcasmBtn') {
            button.innerHTML = '<i class="fas fa-search"></i> ANALYZE SARCASM!';
        } else if (button.id === 'paraphraseBtn') {
            button.innerHTML = '<i class="fas fa-search"></i> CHECK SIMILARITY!';
        } else if (button.id === 'factcheckBtn') {
            button.innerHTML = '<i class="fas fa-search"></i> VERIFY TRUTH!';
        }
    }
}

function showError(resultDiv, message) {
    resultDiv.innerHTML = `
        <div class="alert comic-result comic-error fade-in">
            <div class="comic-error-header">
                <i class="fas fa-exclamation-triangle fa-2x"></i>
                <div class="comic-error-text">
                    <div class="comic-title">💥 OOPS!</div>
                    <div class="comic-subtitle">${message}</div>
                </div>
            </div>
        </div>
    `;
}

// Keep all original API functions unchanged
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        updateSystemStatus(data);
    } catch (error) {
        console.error('Health check failed:', error);
        updateSystemStatus({ status: 'error', models_loaded: {} });
    }
}

async function detectSarcasm() {
    if (isLoading) return;
    
    const text = document.getElementById('sarcasmText').value.trim();
    const button = document.getElementById('sarcasmBtn');
    const resultDiv = document.getElementById('sarcasmResult');
    
    if (!text) {
        showError(resultDiv, 'Please enter some text to analyze! 📝');
        return;
    }
    
    setLoading(button, true);
    
    try {
        const response = await fetch('/api/sarcasm-detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        displaySarcasmResult(result);
        
    } catch (error) {
        showError(resultDiv, `Error: ${error.message}`);
    } finally {
        setLoading(button, false);
    }
}

async function checkParaphrase() {
    if (isLoading) return;
    
    const sentence1 = document.getElementById('sentence1').value.trim();
    const sentence2 = document.getElementById('sentence2').value.trim();
    const button = document.getElementById('paraphraseBtn');
    const resultDiv = document.getElementById('paraphraseResult');
    
    if (!sentence1 || !sentence2) {
        showError(resultDiv, 'Please enter both sentences! ✏️');
        return;
    }
    
    setLoading(button, true);
    
    try {
        const response = await fetch('/api/paraphrase-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sentence1: sentence1, sentence2: sentence2 })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        displayParaphraseResult(result);
        
    } catch (error) {
        showError(resultDiv, `Error: ${error.message}`);
    } finally {
        setLoading(button, false);
    }
}

async function checkFact() {
    if (isLoading) return;
    
    const claim = document.getElementById('claimText').value.trim();
    const button = document.getElementById('factcheckBtn');
    const resultDiv = document.getElementById('factcheckResult');
    
    if (!claim) {
        showError(resultDiv, 'Please enter a claim to verify! 🔍');
        return;
    }
    
    setLoading(button, true);
    
    try {
        const response = await fetch('/api/fact-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ claim: claim })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || 'Request failed');
        }
        
        displayFactCheckResult(result);
        
    } catch (error) {
        showError(resultDiv, `Error: ${error.message}`);
    } finally {
        setLoading(button, false);
    }
}
