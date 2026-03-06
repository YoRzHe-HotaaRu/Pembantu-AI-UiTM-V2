/**
 * UiTM Receptionist AI - Frontend JavaScript
 * Handles chat functionality, theme switching, panel toggling, and streaming responses
 */

// ========================================
// STATE MANAGEMENT
// ========================================

const state = {
    messages: [],
    isTyping: false,
    currentPanel: 'quick', // 'quick' or 'chat' (for mobile)
    theme: localStorage.getItem('uitm-theme') || 'light',
    model: 'google/gemini-3.1-flash-lite-preview',
    currentReasoning: '',
    currentContent: '',
    ragUsed: false,
    structuredResponse: null, // For special responses like creator info
    ttsEnabled: localStorage.getItem('uitm-tts-enabled') !== 'false', // Default: true

    // Audio recording state (for OpenRouter multimodal)
    audio: {
        isRecording: false,
        mediaRecorder: null,
        audioChunks: [],
        stream: null,
        selectedDevice: localStorage.getItem('uitm-selected-mic') || null,
        format: 'webm'
    },

    // VTube Studio state
    vts: {
        enabled: localStorage.getItem('uitm-vts-enabled') === 'true',
        connected: false,
        connecting: false,
        lipSyncData: null,
        currentAudio: null
    },

    // Settings modal state
    settings: {
        isOpen: false
    }
};

// ========================================
// DOM ELEMENTS
// ========================================

const elements = {
    // Settings
    settingsToggle: document.getElementById('settingsToggle'),
    settingsModal: document.getElementById('settingsModal'),
    settingsOverlay: document.getElementById('settingsOverlay'),
    closeSettings: document.getElementById('closeSettings'),
    themeLight: document.getElementById('themeLight'),
    themeDark: document.getElementById('themeDark'),
    microphoneSelect: document.getElementById('microphoneSelect'),
    testMicBtn: document.getElementById('testMicBtn'),
    ttsToggle: document.getElementById('ttsToggle'),
    vtsToggle: document.getElementById('vtsToggle'),
    vtsStatus: document.getElementById('vtsStatus'),
    vtsSettingsSection: document.getElementById('vtsSettingsSection'),
    html: document.documentElement,
    
    // Panels
    quickAccessPanel: document.getElementById('quickAccessPanel'),
    chatPanel: document.getElementById('chatPanel'),
    mobileToggle: document.getElementById('mobileToggle'),
    quickToggleBtn: document.getElementById('quickToggleBtn'),
    chatToggleBtn: document.getElementById('chatToggleBtn'),
    
    // Chat
    messagesArea: document.getElementById('messagesArea'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    charCount: document.getElementById('charCount'),
    
    // Typing indicator
    typingIndicator: document.getElementById('typingIndicator'),
    liveReasoningContainer: document.getElementById('liveReasoningContainer'),
    liveReasoningToggle: document.getElementById('liveReasoningToggle'),
    liveReasoningContent: document.getElementById('liveReasoningContent'),
    liveReasoningText: document.getElementById('liveReasoningText'),
    
    // Quick access items
    quickItems: document.querySelectorAll('.quick-item'),
    
    // Welcome timestamp
    welcomeTime: document.getElementById('welcomeTime'),

    // Voice input
    inputContainer: document.getElementById('inputContainer'),
    micButton: document.getElementById('micButton'),
    inputHint: document.getElementById('inputHint'),
    voiceVisualizer: document.getElementById('voiceVisualizer'),
    sendingVoiceIndicator: document.getElementById('sendingVoiceIndicator')
};

// ========================================
// INITIALIZATION
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Set theme
    setTheme(state.theme);
    
    // Set welcome message timestamp
    elements.welcomeTime.textContent = formatTime(new Date());
    
    // Initialize Lucide icons
    lucide.createIcons();
    
    // Setup event listeners
    setupEventListeners();
    
    // Check screen size for mobile layout
    handleResize();
    
    // Focus input on desktop
    if (window.innerWidth >= 1024) {
        elements.messageInput.focus();
    }

    // Initialize settings
    initializeSettings();

    // Initialize audio devices
    initializeAudioDevices();

    // Initialize VTS (VTube Studio)
    initializeVTS();
    checkVTSStatus();
}

// ========================================
// EVENT LISTENERS
// ========================================

function setupEventListeners() {
    // Settings modal
    elements.settingsToggle.addEventListener('click', toggleSettingsModal);
    elements.closeSettings.addEventListener('click', toggleSettingsModal);
    elements.settingsOverlay.addEventListener('click', toggleSettingsModal);

    // Theme options in settings
    elements.themeLight.addEventListener('click', () => setTheme('light'));
    elements.themeDark.addEventListener('click', () => setTheme('dark'));

    // TTS toggle
    elements.ttsToggle.addEventListener('change', handleTTSToggle);

    // Microphone selection
    elements.microphoneSelect.addEventListener('change', handleMicrophoneSelect);
    elements.testMicBtn.addEventListener('click', testMicrophone);
    
    // Mobile panel toggles
    elements.quickToggleBtn.addEventListener('click', () => switchPanel('quick'));
    elements.chatToggleBtn.addEventListener('click', () => switchPanel('chat'));
    
    // Chat input
    elements.messageInput.addEventListener('input', handleInput);
    elements.messageInput.addEventListener('keydown', handleKeyDown);
    elements.sendButton.addEventListener('click', sendMessage);
    
    // Quick access items
    elements.quickItems.forEach(item => {
        item.addEventListener('click', () => {
            const question = item.getAttribute('data-question');
            handleQuickQuestion(question);
        });
    });
    
    // Live reasoning toggle during typing
    elements.liveReasoningToggle.addEventListener('click', () => {
        elements.liveReasoningContainer.classList.toggle('minimized');
        elements.liveReasoningContainer.classList.toggle('expanded');
    });

    // Voice input toggle
    elements.micButton.addEventListener('click', () => {
        if (state.audio.isRecording) {
            stopAudioRecording();
        } else {
            startAudioRecording();
        }
    });
    
    // Window resize
    window.addEventListener('resize', handleResize);
    
    // Auto-resize textarea
    elements.messageInput.addEventListener('input', autoResizeTextarea);
}

// ========================================
// THEME MANAGEMENT
// ========================================

function setTheme(theme) {
    state.theme = theme;
    elements.html.setAttribute('data-theme', theme);
    localStorage.setItem('uitm-theme', theme);
}

function toggleTheme() {
    const newTheme = state.theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
}

// ========================================
// PANEL MANAGEMENT (Mobile/Tablet)
// ========================================

function switchPanel(panel) {
    state.currentPanel = panel;
    
    // Update toggle buttons
    elements.quickToggleBtn.classList.toggle('active', panel === 'quick');
    elements.chatToggleBtn.classList.toggle('active', panel === 'chat');
    
    // Show/hide panels (mobile/tablet only)
    if (window.innerWidth < 1024) {
        if (panel === 'quick') {
            elements.quickAccessPanel.classList.remove('hidden');
            elements.chatPanel.classList.remove('visible');
        } else {
            elements.quickAccessPanel.classList.add('hidden');
            elements.chatPanel.classList.add('visible');
            // Focus input when switching to chat
            setTimeout(() => elements.messageInput.focus(), 300);
        }
    }
}

function handleResize() {
    const isDesktop = window.innerWidth >= 1024;
    
    if (isDesktop) {
        // Desktop: Show both panels, remove mobile visibility classes
        elements.quickAccessPanel.classList.remove('hidden');
        elements.chatPanel.classList.remove('visible');
        
        // Focus input on desktop
        if (document.activeElement !== elements.messageInput) {
            elements.messageInput.focus();
        }
    } else {
        // Mobile/Tablet: Apply current panel state
        switchPanel(state.currentPanel);
    }
    
    // Update CSS custom property for viewport height (for mobile address bar handling)
    document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
    
    // Dispatch custom event for components that need to know about resize
    window.dispatchEvent(new CustomEvent('app:resize', { 
        detail: { isDesktop, width: window.innerWidth } 
    }));
}

// ========================================
// CHAT INPUT HANDLING
// ========================================

function handleInput() {
    const length = elements.messageInput.value.length;
    elements.charCount.textContent = `${length} / 2000`;
    
    // Update char count color
    if (length > 1800) {
        elements.charCount.style.color = 'var(--accent-primary)';
    } else {
        elements.charCount.style.color = 'var(--text-muted)';
    }
}

function handleKeyDown(e) {
    // Send on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 150) + 'px';
}

// ========================================
// MESSAGE HANDLING
// ========================================

async function sendMessage() {
    const content = elements.messageInput.value.trim();
    if (!content || state.isTyping) return;
    
    // Add user message
    addMessage('user', content);
    
    // Clear input
    elements.messageInput.value = '';
    elements.charCount.textContent = '0 / 2000';
    elements.messageInput.style.height = 'auto';
    
    // Save to state
    state.messages.push({ role: 'user', content });
    
    // Show typing indicator with live reasoning
    showTypingIndicator();
    
    // Send to API
    await sendToAPI();
}

function handleQuickQuestion(question) {
    // Set input value
    elements.messageInput.value = question;
    handleInput();
    
    // Switch to chat panel on mobile
    if (window.innerWidth < 1024) {
        switchPanel('chat');
    }
    
    // Auto-send after short delay
    setTimeout(() => sendMessage(), 300);
}

function addMessage(role, content, reasoning = null, ragUsed = false, imageData = null, isVoice = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const timestamp = formatTime(new Date());
    const isAI = role === 'assistant';
    
    // Build image HTML if present
    let imageHTML = '';
    if (imageData) {
        // Support both direct links in imageData or nested links array
        const links = imageData.links || (imageData._links ? [imageData._links] : []);
        const linksHTML = links.length > 0 ? links.map(link => `
            <a href="${link.url}" target="_blank" rel="noopener" class="creator-link">
                <i data-lucide="${link.icon || 'external-link'}"></i>
                <span>${link.text}</span>
            </a>
        `).join('') : '';
        
        imageHTML = `
            <div class="creator-card">
                <div class="creator-image-wrapper">
                    <img src="${imageData.url}"
                         alt="${imageData.alt || ''}"
                         class="creator-image"
                         loading="lazy">
                </div>
                <div class="creator-info">
                    <h4 class="creator-name">${imageData.title || ''}</h4>
                    ${linksHTML ? `<div class="creator-links">${linksHTML}</div>` : ''}
                </div>
            </div>
        `;
    }
    
    // Build voice visualizer HTML if it's a voice message
    let contentHTML = '';
    if (isVoice) {
        contentHTML = `
            <div class="voice-message-indicator">
                <div class="voice-wave">
                    <span class="voice-bar"></span>
                    <span class="voice-bar"></span>
                    <span class="voice-bar"></span>
                    <span class="voice-bar"></span>
                    <span class="voice-bar"></span>
                </div>
                <span class="voice-label">Mesej Suara</span>
            </div>
        `;
    } else {
        contentHTML = formatMessageContent(content);
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${isAI ? '<img src="/static/assets/bot-avatar.svg" alt="AI Avatar" class="avatar-svg" />' : '<img src="/static/assets/user-avatar.svg" alt="User Avatar" class="avatar-svg" />'}
        </div>
        <div class="message-content">
            <div class="message-header">
                <span class="sender-name">${isAI ? 'AI Receptionist UiTM' : 'Anda'}</span>
                <div class="message-meta">
                    ${ragUsed ? '<span class="rag-indicator" title="Jawapan menggunakan maklumat dari pangkalan pengetahuan UiTM"><i data-lucide="database"></i> RAG</span>' : ''}
                    <span class="timestamp">${timestamp}</span>
                </div>
            </div>
            <div class="message-text ${isVoice ? 'voice-message-text' : ''}">
                ${contentHTML}
            </div>
            ${imageHTML}
            ${reasoning ? createReasoningHTML(reasoning) : ''}
        </div>
    `;
    
    elements.messagesArea.appendChild(messageDiv);
    lucide.createIcons();
    
    // Scroll to bottom
    scrollToBottom();
    
    return messageDiv;
}

function formatMessageContent(content) {
    // Convert markdown-style formatting to HTML
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function createReasoningHTML(reasoning) {
    const reasoningId = 'reasoning-' + Date.now();
    return `
        <div class="reasoning-container minimized" id="${reasoningId}">
            <button class="reasoning-toggle" onclick="toggleReasoning('${reasoningId}')">
                <i data-lucide="brain-circuit"></i>
                <span>Tunjuk Penjelasan AI</span>
                <i data-lucide="chevron-down" class="toggle-icon"></i>
            </button>
            <div class="reasoning-content">
                <div class="reasoning-text">${escapeHtml(reasoning)}</div>
            </div>
        </div>
    `;
}

function toggleReasoning(id) {
    const container = document.getElementById(id);
    if (container) {
        container.classList.toggle('minimized');
        container.classList.toggle('expanded');
        
        // Update button text
        const button = container.querySelector('.reasoning-toggle span');
        if (button) {
            button.textContent = container.classList.contains('expanded') 
                ? 'Sembunyikan Penjelasan' 
                : 'Tunjuk Penjelasan AI';
        }
        
        // Re-render icons
        lucide.createIcons();
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(date) {
    return date.toLocaleTimeString('ms-MY', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

function scrollToBottom() {
    elements.messagesArea.scrollTop = elements.messagesArea.scrollHeight;
}

// ========================================
// TYPING INDICATOR
// ========================================

function showTypingIndicator() {
    state.isTyping = true;
    state.currentReasoning = '';
    state.currentContent = '';
    
    // Show live reasoning expanded by default
    elements.liveReasoningContainer.classList.add('expanded');
    elements.liveReasoningContainer.classList.remove('minimized');
    elements.liveReasoningText.textContent = '';
    
    elements.typingIndicator.style.display = 'block';
    elements.sendButton.disabled = true;
    
    scrollToBottom();
}

function hideTypingIndicator() {
    state.isTyping = false;
    elements.typingIndicator.style.display = 'none';
    elements.sendButton.disabled = false;
    
    // Focus input
    elements.messageInput.focus();
}

function updateLiveReasoning(reasoning) {
    state.currentReasoning = reasoning;
    elements.liveReasoningText.textContent = reasoning;

    // Auto-scroll after DOM update
    requestAnimationFrame(() => {
        // Scroll the reasoning content to show latest thoughts
        elements.liveReasoningContent.scrollTo({
            top: elements.liveReasoningContent.scrollHeight,
            behavior: 'auto'
        });
    });
}

function updateLiveContent(content) {
    state.currentContent = content;
}

// ========================================
// API COMMUNICATION
// ========================================

async function sendToAPI() {
    try {
        // Reset state for new message
        state.currentReasoning = '';
        state.currentContent = '';
        state.ragUsed = false;
        
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: state.messages,
                model: state.model,
                stream: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Process SSE data
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    
                    if (data === '[DONE]') {
                        // Stream complete
                        finalizeResponse();
                        return;
                    }
                    
                    try {
                        const parsed = JSON.parse(data);
                        processStreamData(parsed);
                    } catch (e) {
                        // Ignore parse errors for incomplete chunks
                    }
                }
            }
        }
        
        // Finalize any remaining content
        finalizeResponse();
        
    } catch (error) {
        console.error('API Error:', error);
        hideTypingIndicator();
        addErrorMessage('Maaf, terdapat masalah teknikal. Sila cuba lagi.');
    }
}

function processStreamData(data) {
    // Handle RAG metadata
    if (data.rag_metadata) {
        state.ragUsed = data.rag_metadata.rag_used || false;
        return;
    }
    
    if (!data.choices || !data.choices[0]) return;
    
    const delta = data.choices[0].delta;
    if (!delta) return;
    
    // Handle reasoning (thinking process)
    if (delta.reasoning) {
        updateLiveReasoning(state.currentReasoning + delta.reasoning);
    }
    
    // Handle content (actual response)
    if (delta.content) {
        // Check if this is a special structured response (e.g., creator info)
        try {
            const parsed = JSON.parse(delta.content);
            if (parsed.type === 'creator_info') {
                // Store as structured data for finalizeResponse
                state.structuredResponse = parsed;
                return;
            }
        } catch (e) {
            // Not JSON, treat as regular content
        }
        updateLiveContent(state.currentContent + delta.content);
    }
}

function finalizeResponse() {
    hideTypingIndicator();
    
    // Handle structured response (e.g., creator info with image)
    if (state.structuredResponse) {
        const response = state.structuredResponse;
        
        // Combine image data with links
        const imageData = {
            ...response.image,
            links: response.links || []
        };
        
        addMessage('assistant', response.content, null, false, imageData);
        
        // Play TTS for the response
        playTTS(response.content);
        
        // Save to state
        state.messages.push({
            role: 'assistant',
            content: response.content,
            reasoning: null,
            ragUsed: false,
            image: imageData
        });
        
        // Reset structured response
        state.structuredResponse = null;
        state.ragUsed = false;
        return;
    }
    
    // Add the complete AI message
    if (state.currentContent || state.currentReasoning) {
        const messageContent = state.currentContent || 'Tiada respons.';
        const reasoning = state.currentReasoning || null;
        
        addMessage('assistant', messageContent, reasoning, state.ragUsed);
        
        // Play TTS for the response
        playTTS(messageContent);
        
        // Save to state
        state.messages.push({
            role: 'assistant',
            content: messageContent,
            reasoning: reasoning,
            ragUsed: state.ragUsed
        });
        
        // Reset RAG flag for next message
        state.ragUsed = false;
    }
}

function addErrorMessage(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message ai-message';
    errorDiv.innerHTML = `
        <div class="message-avatar" style="background-color: var(--accent-primary);">
            <i data-lucide="alert-circle"></i>
        </div>
        <div class="message-content" style="border-color: var(--accent-primary);">
            <div class="message-header">
                <span class="sender-name" style="color: var(--accent-primary);">Ralat Sistem</span>
                <span class="timestamp">${formatTime(new Date())}</span>
            </div>
            <div class="message-text" style="color: var(--accent-primary);">
                ${message}
            </div>
        </div>
    `;
    elements.messagesArea.appendChild(errorDiv);
    lucide.createIcons();
    scrollToBottom();
}

// ========================================
// UTILITY FUNCTIONS
// ========================================

// ========================================
// SETTINGS MODULE
// ========================================

function initializeSettings() {
    // Update theme buttons state
    updateThemeButtons();
    // Update TTS toggle state
    updateTTSToggle();
}

function updateTTSToggle() {
    if (elements.ttsToggle) {
        elements.ttsToggle.checked = state.ttsEnabled;
    }
}

function handleTTSToggle(e) {
    state.ttsEnabled = e.target.checked;
    localStorage.setItem('uitm-tts-enabled', state.ttsEnabled);
}

function toggleSettingsModal() {
    state.settings.isOpen = !state.settings.isOpen;
    elements.settingsModal.classList.toggle('open', state.settings.isOpen);

    if (state.settings.isOpen) {
        // Refresh mic list when opening
        populateMicrophoneList();
        updateThemeButtons();
        updateTTSToggle();
    }
}

function updateThemeButtons() {
    // Update active state of theme buttons
    elements.themeLight.classList.toggle('active', state.theme === 'light');
    elements.themeDark.classList.toggle('active', state.theme === 'dark');
}

// Override setTheme to also update buttons
const originalSetTheme = setTheme;
setTheme = function(theme) {
    originalSetTheme(theme);
    updateThemeButtons();
};

// ========================================
// TTS MODULE (Text-to-Speech)
// ========================================

async function playTTS(text) {
    // Skip if TTS is disabled
    if (!state.ttsEnabled) return;
    
    // Skip if no text
    if (!text || text.trim().length === 0) return;
    
    // Clean text for TTS (remove markdown formatting)
    const cleanText = text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/`(.*?)`/g, '$1')
        .replace(/\n/g, ' ')
        .substring(0, 4000); // Limit to 4000 chars
    
    try {
        // Check if VTS is enabled and connected
        const includeLipSync = state.vts.enabled && state.vts.connected;
        
        const response = await fetch('/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: cleanText,
                include_lip_sync: includeLipSync
            })
        });
        
        if (!response.ok) {
            console.error('TTS request failed:', response.status);
            return;
        }
        
        // Check if response is JSON (with lip sync) or audio blob
        const contentType = response.headers.get('content-type');
        
        if (contentType && contentType.includes('application/json')) {
            // Response includes lip sync data
            const data = await response.json();
            
            // Decode base64 audio
            const audioBytes = atob(data.audio);
            const audioArray = new Uint8Array(audioBytes.length);
            for (let i = 0; i < audioBytes.length; i++) {
                audioArray[i] = audioBytes.charCodeAt(i);
            }
            
            const audioBlob = new Blob([audioArray], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            // Store lip sync data
            if (data.lip_sync && data.lip_sync.length > 0) {
                state.vts.lipSyncData = data.lip_sync;
                state.vts.currentAudio = audio;
                
                // Start lip sync playback
                playLipSync(data.lip_sync, audio);
            }
            
            audio.play().catch(err => {
                console.error('Audio playback failed:', err);
            });
            
            // Cleanup URL after playback
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                state.vts.lipSyncData = null;
                state.vts.currentAudio = null;
            };
        } else {
            // Response is audio only
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            
            audio.play().catch(err => {
                console.error('Audio playback failed:', err);
            });
            
            // Cleanup URL after playback
            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
            };
        }
        
    } catch (error) {
        console.error('TTS error:', error);
    }
}

// ========================================
// VTS MODULE (VTube Studio)
// ========================================

async function initializeVTS() {
    // Check if VTS section exists (indicates VTS is enabled on backend)
    if (!elements.vtsSettingsSection) {
        return;
    }
    
    // Update toggle state
    if (elements.vtsToggle) {
        elements.vtsToggle.checked = state.vts.enabled;
        elements.vtsToggle.addEventListener('change', handleVTSToggle);
    }
    
    // If VTS was enabled, try to connect
    if (state.vts.enabled) {
        await connectVTS();
    }
}

async function handleVTSToggle(event) {
    const enabled = event.target.checked;
    state.vts.enabled = enabled;
    localStorage.setItem('uitm-vts-enabled', enabled);
    
    if (enabled) {
        await connectVTS();
    } else {
        await disconnectVTS();
    }
}

async function connectVTS() {
    if (state.vts.connecting || state.vts.connected) return;
    
    state.vts.connecting = true;
    updateVTSStatus('connecting');
    
    try {
        const response = await fetch('/vts/connect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            state.vts.connected = true;
            updateVTSStatus('connected');
            console.log('[VTS] Connected to VTube Studio');
        } else {
            state.vts.connected = false;
            updateVTSStatus('error', data.message || 'Connection failed');
            console.error('[VTS] Connection failed:', data.message);
        }
    } catch (error) {
        state.vts.connected = false;
        updateVTSStatus('error', 'Connection error');
        console.error('[VTS] Connection error:', error);
    } finally {
        state.vts.connecting = false;
    }
}

async function disconnectVTS() {
    if (!state.vts.connected) return;
    
    try {
        await fetch('/vts/disconnect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        state.vts.connected = false;
        updateVTSStatus('disconnected');
        console.log('[VTS] Disconnected from VTube Studio');
    } catch (error) {
        console.error('[VTS] Disconnect error:', error);
    }
}

function updateVTSStatus(status, message = null) {
    if (!elements.vtsStatus) return;
    
    const indicator = elements.vtsStatus.querySelector('.status-indicator');
    const text = elements.vtsStatus.querySelector('.status-text');
    
    if (!indicator || !text) return;
    
    // Remove all status classes
    indicator.classList.remove('connected', 'disconnected', 'connecting', 'error');
    
    switch (status) {
        case 'connected':
            indicator.classList.add('connected');
            text.textContent = 'Disambung';
            break;
        case 'connecting':
            indicator.classList.add('connecting');
            text.textContent = 'Menyambung...';
            break;
        case 'error':
            indicator.classList.add('error');
            text.textContent = message || 'Ralat';
            break;
        default:
            indicator.classList.add('disconnected');
            text.textContent = 'Tidak disambung';
    }
}

async function playLipSync(lipSyncData, audio) {
    if (!lipSyncData || lipSyncData.length === 0) return;
    if (!state.vts.connected) return;
    
    // Send the entire lip sync data to backend to play
    // This ensures proper timing and synchronization
    try {
        await fetch('/vts/play_lip_sync', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ lip_sync: lipSyncData })
        });
    } catch (error) {
        console.error('[VTS] Error playing lip sync:', error);
    }
}

async function sendVTSMouthValue(value) {
    // Send individual mouth value to VTS via backend
    if (!state.vts.connected) return;
    
    try {
        await fetch('/vts/set_mouth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ value: value })
        });
    } catch (error) {
        console.error('[VTS] Error setting mouth:', error);
    }
}

// Check VTS status on page load
async function checkVTSStatus() {
    if (!elements.vtsSettingsSection) return;
    
    try {
        const response = await fetch('/vts/status');
        const data = await response.json();
        
        if (data.enabled) {
            // Backend has VTS enabled, sync frontend state
            // If user previously enabled VTS in frontend, keep that preference
            const userEnabledVTS = localStorage.getItem('uitm-vts-enabled') === 'true';
            
            // Update toggle to reflect user preference
            if (elements.vtsToggle) {
                elements.vtsToggle.checked = userEnabledVTS;
            }
            
            // Update state
            state.vts.enabled = userEnabledVTS;
            state.vts.connected = data.connected;
            
            updateVTSStatus(data.connected ? 'connected' : 'disconnected');
            
            // If user had VTS enabled, try to connect
            if (userEnabledVTS && !data.connected) {
                connectVTS();
            }
        } else {
            // VTS not enabled on backend
            if (elements.vtsSettingsSection) {
                elements.vtsSettingsSection.style.display = 'none';
            }
        }
    } catch (error) {
        console.log('[VTS] Status check failed:', error);
    }
}

// ========================================
// AUDIO DEVICES MODULE
// ========================================

async function initializeAudioDevices() {
    try {
        // Request permission to access microphones
        await navigator.mediaDevices.getUserMedia({ audio: true });

        // Populate the microphone list
        await populateMicrophoneList();
    } catch (error) {
        console.log('Microphone permission not granted or not available:', error);
        elements.microphoneSelect.innerHTML = '<option value="">Tiada kebenaran mikrofon</option>';
    }
}

async function populateMicrophoneList() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');

        // Clear current options
        elements.microphoneSelect.innerHTML = '<option value="">Pilih mikrofon...</option>';

        // Add devices
        audioInputs.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Mikrofon ${index + 1}`;

            // Select saved device
            if (device.deviceId === state.audio.selectedDevice) {
                option.selected = true;
            }

            elements.microphoneSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error enumerating devices:', error);
    }
}

function handleMicrophoneSelect(event) {
    const deviceId = event.target.value;
    state.audio.selectedDevice = deviceId || null;

    if (deviceId) {
        localStorage.setItem('uitm-selected-mic', deviceId);
    } else {
        localStorage.removeItem('uitm-selected-mic');
    }
}

async function testMicrophone() {
    try {
        const constraints = {
            audio: state.audio.selectedDevice
                ? { deviceId: { exact: state.audio.selectedDevice } }
                : true
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        // Visual feedback
        elements.testMicBtn.innerHTML = '<i data-lucide="volume-2"></i> Sedang uji...';
        lucide.createIcons();

        // Stop after 2 seconds
        setTimeout(() => {
            stream.getTracks().forEach(track => track.stop());
            elements.testMicBtn.innerHTML = '<i data-lucide="check"></i> Mikrofon OK';
            lucide.createIcons();

            setTimeout(() => {
                elements.testMicBtn.innerHTML = '<i data-lucide="volume-2"></i> Uji Mikrofon';
                lucide.createIcons();
            }, 2000);
        }, 2000);
    } catch (error) {
        console.error('Microphone test failed:', error);
        elements.testMicBtn.innerHTML = '<i data-lucide="x"></i> Ralat';
        lucide.createIcons();

        setTimeout(() => {
            elements.testMicBtn.innerHTML = '<i data-lucide="volume-2"></i> Uji Mikrofon';
            lucide.createIcons();
        }, 2000);
    }
}

// ========================================
// AUDIO RECORDING MODULE
// ========================================

async function startAudioRecording() {
    try {
        // Get microphone stream
        const constraints = {
            audio: state.audio.selectedDevice
                ? { deviceId: { exact: state.audio.selectedDevice } }
                : true
        };

        state.audio.stream = await navigator.mediaDevices.getUserMedia(constraints);

        // Create MediaRecorder
        state.audio.mediaRecorder = new MediaRecorder(state.audio.stream, {
            mimeType: 'audio/webm'
        });

        state.audio.audioChunks = [];

        // Handle data available
        state.audio.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audio.audioChunks.push(event.data);
            }
        };

        // Handle recording stop
        state.audio.mediaRecorder.onstop = handleRecordingStop;

        // Start recording
        state.audio.mediaRecorder.start(100); // Collect data every 100ms
        state.audio.isRecording = true;

        // Update UI
        enterRecordingMode();

        // Auto-stop after 30 seconds (prevent long recordings)
        setTimeout(() => {
            if (state.audio.isRecording) {
                stopAudioRecording();
            }
        }, 30000);

    } catch (error) {
        console.error('Error starting audio recording:', error);
        elements.inputHint.textContent = 'Tidak dapat mengakses mikrofon. Pastikan kebenaran diberi.';
    }
}

function stopAudioRecording() {
    if (state.audio.mediaRecorder && state.audio.mediaRecorder.state !== 'inactive') {
        state.audio.mediaRecorder.stop();
    }

    if (state.audio.stream) {
        state.audio.stream.getTracks().forEach(track => track.stop());
    }

    state.audio.isRecording = false;
}

async function handleRecordingStop() {
    if (state.audio.audioChunks.length === 0) {
        exitRecordingMode();
        return;
    }

    // Create blob from chunks
    const audioBlob = new Blob(state.audio.audioChunks, { type: 'audio/webm' });

    // Convert to base64
    const base64Audio = await blobToBase64(audioBlob);

    // Show sending state
    elements.inputHint.textContent = 'Menghantar audio...';

    // Send as multimodal message
    await sendAudioMessage(base64Audio);

    // Clear chunks
    state.audio.audioChunks = [];

    exitRecordingMode();
}

function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            // Get base64 string without data URL prefix
            const base64String = reader.result.split(',')[1];
            resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function showSendingVoiceState() {
    elements.inputContainer.classList.remove('voice-mode');
    elements.inputContainer.classList.add('sending-voice');
    elements.micButton.classList.remove('active');
    elements.inputHint.textContent = 'Menghantar audio ke AI...';
}

function hideSendingVoiceState() {
    elements.inputContainer.classList.remove('sending-voice');
    elements.inputHint.textContent = 'Tekan Enter untuk hantar, Shift+Enter untuk baris baru';
}

async function sendAudioMessage(base64Audio) {
    // Show sending state
    showSendingVoiceState();

    // Disable input while sending
    elements.sendButton.disabled = true;

    // Add user message to UI with voice indicator
    addMessage('user', '[Mesej Suara]', null, false, null, true);

    // Prepare multimodal message
    const multimodalMessage = {
        role: 'user',
        content: [
            {
                type: 'text',
                text: 'Sila transkripsi mesej suara ini dan jawab.'
            },
            {
                type: 'input_audio',
                input_audio: {
                    data: base64Audio,
                    format: 'webm'
                }
            }
        ]
    };

    // Send to API
    try {
        await sendToAPIWithMessage(multimodalMessage);
    } catch (error) {
        console.error('Error sending audio message:', error);
        addErrorMessage('Ralat menghantar mesej audio. Sila cuba lagi.');
    }

    // Hide sending state
    hideSendingVoiceState();
    elements.sendButton.disabled = false;
}

// Modified sendToAPI that accepts a pre-built message
async function sendToAPIWithMessage(message) {
    try {
        state.isTyping = true;
        showTypingIndicator();

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                messages: [...state.messages, message],
                model: state.model,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        state.currentReasoning = '';
        state.currentContent = '';
        state.ragUsed = false;  // Reset RAG flag for new message

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                const dataStr = line.slice(6).trim();
                if (dataStr === '[DONE]') continue;

                try {
                    const data = JSON.parse(dataStr);
                    processStreamData(data);
                } catch (e) {
                    // Ignore parse errors
                }
            }
        }

        finalizeResponse();

    } catch (error) {
        console.error('API Error:', error);
        addErrorMessage('Maaf, terdapat ralat semasa berkomunikasi dengan AI. Sila cuba lagi.');
        finalizeResponse();
    }
}

function enterRecordingMode() {
    elements.inputContainer.classList.add('voice-mode');
    elements.micButton.classList.add('active');
    elements.inputHint.textContent = 'Merakam... ketuk mic untuk berhenti';
    elements.messageInput.placeholder = 'Merakam audio...';
}

function exitRecordingMode() {
    elements.inputContainer.classList.remove('voice-mode');
    elements.micButton.classList.remove('active');
    elements.inputHint.textContent = 'Tekan Enter untuk hantar, Shift+Enter untuk baris baru';
    elements.messageInput.placeholder = 'Taip mesej anda di sini...';
}

// Expose toggleReasoning to global scope for onclick handlers
window.toggleReasoning = toggleReasoning;

// Handle page visibility change (pause/resume functionality)
document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.isTyping) {
        // Page hidden while typing - could add pause functionality here
        console.log('Page hidden - chat continues in background');
    }
});

// Prevent accidental page leave while typing
window.addEventListener('beforeunload', (e) => {
    if (state.isTyping) {
        e.preventDefault();
        e.returnValue = 'AI sedang menjawab. Anda pasti mahu tinggalkan halaman?';
    }
});
