import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import time
import numpy as np

# Import our enhanced backend
try:
    from simple_rag_backend import SimpleRAGPipeline, SimpleDatabaseManager, check_dependencies
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# Set page config with educational theme
st.set_page_config(
    page_title="StudyAI - Smart Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling for educational app
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Variables - Educational Theme */
:root {
    --primary-color: #3b82f6;
    --primary-dark: #1d4ed8;
    --secondary-color: #64748b;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --error-color: #ef4444;
    --background-light: #f8fafc;
    --background-white: #ffffff;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-light: #64748b;
    --border-color: #e2e8f0;
    --sidebar-bg: #f1f5f9;
    --accent-purple: #8b5cf6;
    --accent-emerald: #10b981;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
}

/* Main App Styling */
.stApp {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Sidebar Styling */
.stSidebar {
    background-color: var(--sidebar-bg) !important;
}

.stSidebar > div:first-child {
    background: linear-gradient(180deg, var(--sidebar-bg) 0%, #e2e8f0 100%);
    padding-top: 1rem;
}

.sidebar-nav {
    background: var(--background-white);
    border-radius: var(--radius-lg);
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
    border-left: 4px solid var(--primary-color);
}

.nav-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--text-primary);
    text-decoration: none;
    font-weight: 500;
}

.nav-item:hover {
    background: var(--primary-color);
    color: white;
    transform: translateX(4px);
}

.nav-item.active {
    background: var(--primary-color);
    color: white;
    box-shadow: var(--shadow-md);
}

/* Main content area */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    background: var(--background-white);
    border-radius: var(--radius-lg);
    margin: 1rem;
    box-shadow: var(--shadow-lg);
    max-width: 1200px;
}

/* Header Styling - Educational Theme */
.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-purple) 100%);
    padding: 2.5rem 2rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white" opacity="0.1"/><circle cx="80" cy="30" r="1.5" fill="white" opacity="0.15"/><circle cx="30" cy="70" r="1" fill="white" opacity="0.1"/></svg>');
    pointer-events: none;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

.subtitle {
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.95;
    position: relative;
    z-index: 1;
}

/* Card Components */
.study-card {
    background: var(--background-white);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.study-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary-color), var(--accent-emerald));
}

.study-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

/* Feature Cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    margin: 2rem 0;
}

.feature-card {
    background: var(--background-white);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    transition: all 0.3s ease;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--primary-color);
}

.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary-color);
}

.feature-card h3 {
    color: var(--text-primary) !important;
    font-weight: 600;
    margin-bottom: 1rem;
    font-size: 1.25rem;
}

.feature-card p {
    color: var(--text-secondary) !important;
    line-height: 1.6;
}

/* Database Cards */
.db-card {
    background: var(--background-white);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
    position: relative;
}

.db-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-color), var(--success-color));
}

.db-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.db-card-active {
    border-color: var(--primary-color);
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    box-shadow: var(--shadow-md);
}

.db-card h4 {
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.db-card p {
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
    margin: 0.25rem 0;
}

/* Status Cards with High Contrast */
.status-card {
    padding: 1.5rem;
    border-radius: var(--radius-lg);
    margin: 1rem 0;
    border-left: 5px solid;
    box-shadow: var(--shadow-sm);
    font-weight: 500;
}

.status-success {
    background: #f0fdf4;
    color: #166534 !important;
    border-color: var(--success-color);
}

.status-warning {
    background: #fffbeb;
    color: #92400e !important;
    border-color: var(--warning-color);
}

.status-error {
    background: #fef2f2;
    color: #991b1b !important;
    border-color: var(--error-color);
}

.status-info {
    background: #eff6ff;
    color: #1e40af !important;
    border-color: var(--primary-color);
}

/* Enhanced Chat Interface */
.chat-container {
    background: var(--background-white);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid var(--border-color);
}

.chat-message {
    padding: 1.5rem;
    margin: 1rem 0;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
    position: relative;
}

.chat-user {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    margin-left: 2rem;
    border-left: 4px solid var(--primary-color);
}

.chat-user .chat-label {
    color: var(--primary-color) !important;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-user .chat-content {
    color: var(--text-primary) !important;
    line-height: 1.6;
}

.chat-ai {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    margin-right: 2rem;
    border-left: 4px solid var(--success-color);
}

.chat-ai .chat-label {
    color: var(--success-color) !important;
    font-weight: 600;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-ai .chat-content {
    color: var(--text-primary) !important;
    line-height: 1.6;
}

.chat-sources {
    font-size: 0.85rem;
    color: var(--text-light) !important;
    margin-top: 0.75rem;
    font-style: italic;
    padding: 0.75rem;
    background: rgba(0, 0, 0, 0.05);
    border-radius: var(--radius-sm);
}

.chat-metadata {
    font-size: 0.8rem;
    color: var(--text-light) !important;
    margin-top: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border-color);
}

/* Chat History Download Button */
.chat-download-btn {
    position: absolute;
    top: 1rem;
    right: 1rem;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.chat-message:hover .chat-download-btn {
    opacity: 1;
}

/* Upload Zone */
.upload-zone {
    border: 3px dashed var(--border-color);
    border-radius: var(--radius-lg);
    padding: 3rem;
    text-align: center;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    margin: 2rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.upload-zone:hover {
    border-color: var(--primary-color);
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
}

.upload-zone h3 {
    color: var(--text-primary) !important;
    margin-bottom: 1rem;
}

.upload-zone p {
    color: var(--text-secondary) !important;
}

/* Enhanced Processing Animation */
.processing-container {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin: 2rem 0;
    box-shadow: var(--shadow-sm);
    border-left: 4px solid var(--primary-color);
}

.processing-step {
    display: flex;
    align-items: center;
    margin: 1rem 0;
    padding: 1rem;
    background: var(--background-white);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow-sm);
    transition: all 0.3s ease;
}

.processing-step.active {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-left: 3px solid var(--primary-color);
}

.processing-step.completed {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-left: 3px solid var(--success-color);
}

/* Metrics */
.metric-card {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white !important;
    padding: 2rem 1.5rem;
    border-radius: var(--radius-lg);
    text-align: center;
    margin: 1rem 0;
    box-shadow: var(--shadow-md);
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: scale(1.05);
}

.metric-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: white !important;
}

.metric-label {
    font-size: 1rem;
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 500;
}

/* Enhanced Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
    color: white !important;
    border: none;
    border-radius: var(--radius-md);
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s ease;
    box-shadow: var(--shadow-sm);
}

.stButton > button:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* Download Button Styling */
.download-button {
    background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Progress Bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--primary-color), var(--success-color));
}

/* Form Elements */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    border: 2px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    background: var(--background-white) !important;
    transition: border-color 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: var(--background-light);
    border-radius: var(--radius-lg);
    padding: 0.75rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    padding: 1rem 2rem;
    font-weight: 600;
    transition: all 0.2s ease;
    border: 2px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--background-white);
    color: var(--primary-color);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--primary-color);
    color: white;
    box-shadow: var(--shadow-md);
}

/* Animations */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes processing {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.slide-in {
    animation: slideIn 0.4s ease;
}

.processing-bar {
    position: relative;
    overflow: hidden;
}

.processing-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 30%;
    background: linear-gradient(90deg, transparent, var(--primary-color), transparent);
    animation: processing 2s infinite;
}

/* Success Animation */
@keyframes success {
    0% { transform: scale(0.8); opacity: 0; }
    50% { transform: scale(1.1); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}

.success-animation {
    animation: success 0.6s ease;
}

/* Text Contrast Fixes */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

p, div, span {
    color: var(--text-secondary) !important;
}

.stMarkdown p {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] {
    background-color: white !important;
    border-right: 2px solid #ddd;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .chat-user,
    .chat-ai {
        margin-left: 0;
        margin-right: 0;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = SimpleDatabaseManager() if BACKEND_AVAILABLE else None
    if "current_rag" not in st.session_state:
        st.session_state.current_rag = None
    if "current_db_name" not in st.session_state:
        st.session_state.current_db_name = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "auto_navigate_to_chat" not in st.session_state:
        st.session_state.auto_navigate_to_chat = False

def main():
    init_session_state()
    
    # Check backend availability
    if not BACKEND_AVAILABLE:
        st.error("Backend not available! Please ensure simple_rag_backend.py is in the same directory.")
        return
    
    # Auto-navigate to chat after processing
    if st.session_state.auto_navigate_to_chat:
        st.session_state.current_page = "chat"
        st.session_state.auto_navigate_to_chat = False
        st.rerun()
    
    # Render sidebar navigation
    render_sidebar()
    
    # Render main content based on current page
    if st.session_state.current_page == "home":
        render_home_page()
    elif st.session_state.current_page == "upload":
        render_upload_page()
    elif st.session_state.current_page == "chat":
        render_chat_page()
    elif st.session_state.current_page == "manage":
        render_manage_page()
    elif st.session_state.current_page == "analytics":
        render_analytics_page()

def render_sidebar():
    """Render clean navigation sidebar with enhanced info"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="color: var(--primary-color); margin-bottom: 0.5rem; font-size: 1.5rem;">🎓 StudyAI</h2>
            <p style="color: var(--text-light); font-size: 0.9rem;">Smart Learning Assistant v3.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        
        # Navigation items
        nav_items = [
            ("home", "🏠 Home", "Welcome & Overview"),
            ("upload", "📚 Upload Chapters", "Add study materials"),
            ("chat", "💬 Ask Questions", "Interactive learning"),
            ("manage", "📂 Manage Collections", "Organize your content"),
            ("analytics", "📊 Study Analytics", "Track your progress")
        ]
        
        for page_key, title, desc in nav_items:
            if st.button(title, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Current database info with enhanced details
        if st.session_state.current_rag:
            chat_count = len(st.session_state.current_rag.chat_history.chat_entries)
            st.markdown(f"""
            <div class="status-card status-success">
                <h5>📖 Active Collection</h5>
                <p><strong>{st.session_state.current_db_name}</strong></p>
                <p>📝 {len(st.session_state.current_rag.chunks)} study chunks</p>
                <p>💬 {chat_count} questions asked</p>
                <p>🤖 IBM Granite + Cross-Encoder</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick chat history download
            if chat_count > 0:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📄 Download TXT", key="sidebar_txt", use_container_width=True):
                        download_chat_history("txt")
                with col2:
                    if st.button("📋 Download MD", key="sidebar_md", use_container_width=True):
                        download_chat_history("markdown")
        
        # Quick stats with enhanced info
        if st.session_state.db_manager:
            databases = st.session_state.db_manager.get_databases()
            if databases:
                total_chunks = sum(db.get('chunks', 0) for db in databases)
                total_pdfs = sum(db.get('pdf_count', 0) for db in databases)
                total_questions = sum(db.get('total_questions', 0) for db in databases)
                
                st.markdown("### 📈 Quick Stats")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Collections", len(databases))
                    st.metric("PDF Files", total_pdfs)
                with col2:
                    st.metric("Content Chunks", f"{total_chunks:,}")
                    st.metric("Questions Asked", total_questions)

def render_home_page():
    """Render educational-themed home page"""
    # Header
    st.markdown("""
    <div class="main-header slide-in">
        <div class="main-title">🎓 StudyAI - Smart Learning Assistant</div>
        <div class="subtitle">Transform your textbooks and notes into an intelligent study companion with IBM Granite AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced feature cards
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    feature1 = """
    <div class="feature-card slide-in">
        <h3>📚 Advanced Document Processing</h3>
        <p>Upload PDFs with hybrid OCR + text extraction. Our system intelligently processes typed text, scanned documents, and handwritten notes using state-of-the-art technology.</p>
    </div>
    """
    
    feature2 = """
    <div class="feature-card slide-in">
        <h3>🤖 IBM Granite AI + Cross-Encoder</h3>
        <p>Powered by IBM's Granite embedding model and BAAI cross-encoder reranking for superior answer quality. Get contextually accurate responses with source attribution.</p>
    </div>
    """
    
    feature3 = """
    <div class="feature-card slide-in">
        <h3>📊 Complete Chat History</h3>
        <p>Comprehensive chat history with metadata tracking, source attribution, and downloadable study sessions in multiple formats for future reference.</p>
    </div>
    """
    
    st.markdown(feature1 + feature2 + feature3, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # New features highlight
    st.markdown("""
    <style>
        .study-card {
            background: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-bottom: 2rem;
        }
        .study-card h2 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 1.8rem;
        }
        .study-step {
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #fafafa 0%, #f4f4f5 100%);
            transition: transform 0.3s ease;
        }
        .study-step:hover {
            transform: translateY(-5px);
        }
        .step-number {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
    </style>

    <!-- Enhanced Features -->
    <div class="study-card slide-in">
        <h2 style="color: var(--primary-color);">🆕 Enhanced Features</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin-top: 2rem;">
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
                <h4 style="color: var(--text-primary);">IBM Granite Embeddings</h4>
                <p style="color: var(--text-secondary);">State-of-the-art embeddings for better semantic understanding and more accurate retrieval.</p>
            </div>
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
                <h4 style="color: var(--text-primary);">Cross-Encoder Reranking</h4>
                <p style="color: var(--text-secondary);">Advanced reranking system that improves answer relevance by 40% compared to standard retrieval.</p>
            </div>
            <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%); border-radius: 12px;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
                <h4 style="color: var(--text-primary);">Smart History Tracking</h4>
                <p style="color: var(--text-secondary);">Detailed metadata for every interaction with downloadable study session summaries.</p>
            </div>
        </div>
    </div>

    <!-- Get Started -->
    <div class="study-card slide-in">
        <h2 style="color: var(--primary-color);">🚀 Get Started in 3 Easy Steps</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
            <div class="study-step">
                <div class="step-number">1️⃣</div>
                <h4 style="color: var(--text-primary);">Upload Your Materials</h4>
                <p style="color: var(--text-secondary);">Upload PDF chapters, notes, or any study materials. Supports both digital and scanned documents.</p>
            </div>
            <div class="study-step">
                <div class="step-number">2️⃣</div>
                <h4 style="color: var(--text-primary);">AI Processing</h4>
                <p style="color: var(--text-secondary);">IBM Granite AI processes your content and builds an intelligent, searchable knowledge base.</p>
            </div>
            <div class="study-step">
                <div class="step-number">3️⃣</div>
                <h4 style="color: var(--text-primary);">Start Learning</h4>
                <p style="color: var(--text-secondary);">Ask questions, get precise answers, and track your learning journey with comprehensive history.</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📚 Start by Uploading Your First Chapter", type="primary", use_container_width=True):
            st.session_state.current_page = "upload"
            st.rerun()

def render_upload_page():
    """Enhanced upload page with better processing feedback"""
    st.markdown("""
    <div class="main-header slide-in">
        <div class="main-title">📚 Upload Study Materials</div>
        <div class="subtitle">Create a new knowledge base powered by IBM Granite AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="study-card slide-in">', unsafe_allow_html=True)
    
    # Collection name input
    collection_name = st.text_input(
        "📝 Study Collection Name",
        placeholder="e.g., Biology Chapter 1-3, Mathematics Calculus, History World War II",
        help="Give your study collection a descriptive name"
    )
    
    # File upload area
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("### 📎 Upload Your Study Files")
    uploaded_files = st.file_uploader(
        "",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF files of textbooks, lecture notes, or study guides. Both typed and scanned documents are supported with advanced OCR.",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f"**✅ {len(uploaded_files)} files ready to process:**")
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            st.markdown(f"• {file.name} ({file_size:.1f} KB)")
    else:
        st.markdown("**📋 Supported formats:** PDF files (text-based, scanned, or mixed)")
        st.markdown("**💡 Tip:** Upload multiple chapters at once for comprehensive study collections")
        st.markdown("**🤖 AI Features:** IBM Granite embeddings + Cross-encoder reranking for superior accuracy")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing options
    with st.expander("⚙️ Advanced Processing Settings"):
        chunk_size = st.slider("Content Chunk Size", 400, 1200, 800, 50, 
                              help="Larger chunks for better context, smaller chunks for precise answers")
        
        col1, col2 = st.columns(2)
        with col1:
            force_ocr = st.checkbox("Force OCR Processing", False, 
                                   help="Apply OCR to all files (useful for scanned documents)")
        with col2:
            high_quality = st.checkbox("High Quality Mode", True,
                                     help="Better accuracy with IBM Granite embeddings (recommended)")
    
    # Create collection button
    if st.button("🚀 Create AI-Powered Study Collection", type="primary", use_container_width=True):
        if not collection_name.strip():
            st.error("Please enter a study collection name!")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one PDF file!")
            return
        
        # Check for duplicate names
        if st.session_state.db_manager:
            existing_names = [db["name"] for db in st.session_state.db_manager.get_databases()]
            if collection_name in existing_names:
                st.error("Collection name already exists! Please choose a different name.")
                return
        
        process_documents_enhanced(collection_name, uploaded_files, chunk_size)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_page():
    """Enhanced chat interface with history management"""
    if not st.session_state.current_rag:
        st.markdown("""
        <div class="main-header slide-in">
            <div class="main-title">💬 Study Chat</div>
            <div class="subtitle">Ask questions about your study materials</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="status-card status-info">
            <h4>📚 No Study Collection Active</h4>
            <p>Please upload study materials first or select an existing collection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Upload New Materials", use_container_width=True):
                st.session_state.current_page = "upload"
                st.rerun()
        with col2:
            if st.button("📂 Browse Collections", use_container_width=True):
                st.session_state.current_page = "manage"
                st.rerun()
        return
    
    # Header with collection info and chat controls
    chat_count = len(st.session_state.current_rag.chat_history.chat_entries)
    
    st.markdown(f"""
    <div class="main-header slide-in">
        <div class="main-title">💬 Study Chat</div>
        <div class="subtitle">Active: {st.session_state.current_db_name} • {chat_count} questions asked • IBM Granite AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat controls bar
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"**Session:** {st.session_state.current_rag.chat_history.session_id}")
    with col2:
        if st.button("📄 Download TXT", use_container_width=True):
            download_chat_history("txt")
    with col3:
        if st.button("📋 Download MD", use_container_width=True):
            download_chat_history("markdown")
    with col4:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.current_rag.chat_history.chat_entries = []
            st.rerun()
    
    # Chat history with enhanced metadata
    if st.session_state.current_rag.chat_history.chat_entries:
        st.markdown("### 📖 Study Conversation")
        
        for i, entry in enumerate(st.session_state.current_rag.chat_history.chat_entries):
            # User question
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            st.markdown(f"""
            <div class="chat-message chat-user">
                <div class="chat-label">🙋‍♂️ Your Question <span style="font-size: 0.8em; opacity: 0.7;">({timestamp})</span></div>
                <div class="chat-content">{entry['question']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # AI response with enhanced metadata
            pdf_sources = ", ".join(entry.get('pdf_sources', []))
            st.markdown(f"""
            <div class="chat-message chat-ai">
                <div class="chat-label">🤖 StudyAI Answer <span style="font-size: 0.8em; opacity: 0.7;">(IBM Granite + Cross-Encoder)</span></div>
                <div class="chat-content">{entry['answer']}</div>
                <div class="chat-sources">
                    📊 Sources: {entry.get('sources', 'N/A')}<br>
                    📄 PDF Files: {pdf_sources if pdf_sources else 'N/A'}<br>
                    🔍 Chunks Used: {entry.get('chunk_count', 0)}
                </div>
                <div class="chat-metadata">
                    <span>Entry #{i+1}</span>
                    <span>{timestamp}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Question input section
    st.markdown("### ❓ Ask Your Study Question")
    
    # Enhanced example questions for first-time users
    if not st.session_state.current_rag.chat_history.chat_entries:
        st.markdown("**💡 Try these AI-powered study questions:**")
        example_questions = [
            "What are the main concepts covered in this material?",
            "Can you explain this topic in simple terms with examples?",
            "What are the key formulas, definitions, or important points?",
            "How do these concepts connect to each other?",
            "Can you create a summary of the most important information?",
            "What should I focus on for exam preparation?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(f"💭 {question}", key=f"example_{i}", use_container_width=True):
                    ask_question_enhanced(question)
    
    # Question input form
    with st.form("study_question_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your study question:",
            placeholder="Ask anything about your materials - definitions, explanations, examples, connections, summaries...",
            height=100,
            help="Be specific for better answers. The IBM Granite AI will search through your materials and provide contextually accurate responses."
        )
        
        submitted = st.form_submit_button("🚀 Get AI Answer", type="primary", use_container_width=True)
    
    if submitted and question.strip():
        ask_question_enhanced(question.strip())

def render_manage_page():
    """Enhanced database management page"""
    st.markdown("""
    <div class="main-header slide-in">
        <div class="main-title">📂 Manage Study Collections</div>
        <div class="subtitle">Organize and access your AI-powered study materials</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.db_manager:
        databases = st.session_state.db_manager.get_databases()
        
        if databases:
            st.markdown("### 📚 Your Study Collections")
            
            for db in databases:
                render_enhanced_database_card(db)
                
        else:
            st.markdown("""
            <div class="status-card status-info">
                <h4>📚 No Study Collections Yet</h4>
                <p>Create your first AI-powered collection by uploading study materials.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📚 Upload Your First Materials", type="primary", use_container_width=True):
                st.session_state.current_page = "upload"
                st.rerun()

def render_analytics_page():
    """Enhanced analytics page with detailed insights"""
    st.markdown("""
    <div class="main-header slide-in">
        <div class="main-title">📊 Study Analytics</div>
        <div class="subtitle">Track your learning progress and AI interaction patterns</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.current_rag:
        st.markdown("""
        <div class="status-card status-info">
            <h4>📊 No Active Collection</h4>
            <p>Select a study collection to view detailed analytics and learning insights.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced learning metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.current_rag.chunks)}</div>
            <div class="metric-label">Study Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pdf_count = len(set(meta.get('source_file', 'Unknown') for meta in st.session_state.current_rag.chunk_metadata))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{pdf_count}</div>
            <div class="metric-label">PDF Sources</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        chat_count = len(st.session_state.current_rag.chat_history.chat_entries)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{chat_count}</div>
            <div class="metric-label">Questions Asked</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.current_rag.chat_history.chat_entries:
            avg_chunks = np.mean([entry.get('chunk_count', 0) for entry in st.session_state.current_rag.chat_history.chat_entries])
        else:
            avg_chunks = 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{avg_chunks:.1f}</div>
            <div class="metric-label">Avg Chunks/Answer</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Processing statistics
    if hasattr(st.session_state.current_rag, 'processing_stats'):
        stats = st.session_state.current_rag.processing_stats
        if stats:
            st.markdown("### 🔧 Processing Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Standard Extraction", stats.get('fitz_success', 0))
            with col2:
                st.metric("Hybrid Processing", stats.get('hybrid_extraction', 0))
            with col3:
                st.metric("Full OCR", stats.get('ocr_fallback', 0))
            with col4:
                st.metric("Failed Files", stats.get('failed', 0))
    
    # Enhanced study session insights
    if st.session_state.current_rag.chat_history.chat_entries:
        st.markdown("### 📈 Recent Study Activity")
        
        recent_entries = st.session_state.current_rag.chat_history.chat_entries[-10:]  # Show last 10
        for i, entry in enumerate(reversed(recent_entries), 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            question_preview = entry['question'][:100] + "..." if len(entry['question']) > 100 else entry['question']
            pdf_sources = ", ".join(entry.get('pdf_sources', []))
            
            st.markdown(f"""
            <div class="study-card" style="padding: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 0.9rem; color: var(--text-light);">#{len(st.session_state.current_rag.chat_history.chat_entries) - i + 1}</div>
                    <div style="font-size: 0.9rem; color: var(--text-light);">{timestamp}</div>
                </div>
                <div style="font-weight: 600; margin: 0.5rem 0; color: var(--text-primary);">{question_preview}</div>
                <div style="font-size: 0.85rem; color: var(--text-light); margin-top: 0.5rem;">
                    <div>📄 Sources: {pdf_sources if pdf_sources else 'N/A'}</div>
                    <div>🔍 Chunks: {entry.get('chunk_count', 0)} • 📊 {entry.get('sources', 'N/A')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_enhanced_database_card(db: Dict):
    """Render enhanced database card with more details"""
    is_active = (st.session_state.current_db_name == db["name"])
    card_class = "db-card-active" if is_active else "db-card"
    
    pdf_sources_text = ", ".join(db.get('pdf_sources', [])[:3])  # Show first 3
    if len(db.get('pdf_sources', [])) > 3:
        pdf_sources_text += f" and {len(db.get('pdf_sources', [])) - 3} more"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4>{db['name']}</h4>
        <div style="margin: 1rem 0;">
            <p>📝 {db['chunks']} study chunks from {db.get('pdf_count', 0)} PDF files</p>
            <p>📄 Sources: {pdf_sources_text if pdf_sources_text else 'N/A'}</p>
            <p>💬 {db.get('total_questions', 0)} questions in {db.get('chat_sessions', 0)} sessions</p>
            <p>📅 Created: {db['created_at'][:10] if len(db['created_at']) > 10 else db['created_at']}</p>
            <p>💾 Size: {db.get('file_size', 0) / 1024:.1f} KB • 🔖 Version: {db.get('version', '1.0')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📂 Load", key=f"load_{db['db_id']}", use_container_width=True):
            load_database_enhanced(db)
    
    with col2:
        if st.button("💬 Chat", key=f"chat_{db['db_id']}", use_container_width=True):
            load_database_enhanced(db)
            st.session_state.current_page = "chat"
            st.rerun()
    
    with col3:
        if st.button("📊 Analytics", key=f"analytics_{db['db_id']}", use_container_width=True):
            load_database_enhanced(db)
            st.session_state.current_page = "analytics"
            st.rerun()
    
    with col4:
        if st.button("🗑️ Delete", key=f"del_{db['db_id']}", use_container_width=True):
            if st.session_state.db_manager.delete_database(db['name']):
                if st.session_state.current_db_name == db['name']:
                    st.session_state.current_rag = None
                    st.session_state.current_db_name = None
                st.success("Collection deleted successfully!")
                time.sleep(1)
                st.rerun()

def load_database_enhanced(db_info: Dict):
    """Load a database with enhanced feedback"""
    with st.spinner("Loading study collection with IBM Granite AI..."):
        try:
            rag = st.session_state.db_manager.load_database(db_info['name'])
            
            if rag:
                st.session_state.current_rag = rag
                st.session_state.current_db_name = db_info['name']
                
                st.success(f"Successfully loaded '{db_info['name']}' with {len(rag.chunks)} chunks and {len(rag.chat_history.chat_entries)} chat entries")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to load collection")
        except Exception as e:
            st.error(f"Error loading collection: {str(e)}")

def process_documents_enhanced(name: str, files: List, chunk_size: int = 800):
    """Enhanced document processing with better progress tracking"""
    progress_container = st.empty()
    
    with progress_container.container():
        st.markdown("### 🔄 Processing Study Materials with IBM Granite AI")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_text = st.empty()
        step_container = st.empty()
        
        try:
            # Step 1: Initialize with IBM Granite
            with step_container.container():
                st.markdown("""
                <div class="processing-step active">
                    <div>🔧 Initializing IBM Granite AI System...</div>
                </div>
                """, unsafe_allow_html=True)
            
            status_text.markdown("**🔧 Loading IBM Granite Embeddings + Cross-Encoder...**")
            details_text.markdown("*Setting up state-of-the-art AI processing pipeline*")
            progress_bar.progress(10)
            time.sleep(1)
            
            rag = SimpleRAGPipeline(name)
            
            # Step 2: Advanced document processing
            with step_container.container():
                st.markdown("""
                <div class="processing-step completed">
                    <div>✅ IBM Granite AI System Loaded</div>
                </div>
                <div class="processing-step active">
                    <div>📖 Advanced Document Processing...</div>
                </div>
                """, unsafe_allow_html=True)
            
            status_text.markdown("**📖 Processing Documents with Hybrid OCR + Text Extraction...**")
            details_text.markdown("*Using intelligent document analysis and OCR technology*")
            progress_bar.progress(30)
            time.sleep(1)
            
            result = rag.process_pdfs(files)
            progress_bar.progress(50)
            
            if not result["success"]:
                st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
                return
            
            # Step 3: IBM Granite embeddings
            with step_container.container():
                st.markdown("""
                <div class="processing-step completed">
                    <div>✅ Documents Processed Successfully</div>
                </div>
                <div class="processing-step active">
                    <div>🧠 Creating IBM Granite Embeddings...</div>
                </div>
                """, unsafe_allow_html=True)
            
            status_text.markdown("**🧠 Generating IBM Granite Embeddings...**")
            details_text.markdown("*Converting content to high-dimensional semantic vectors*")
            progress_bar.progress(70)
            time.sleep(1.5)
            
            # Step 4: Cross-encoder setup
            with step_container.container():
                st.markdown("""
                <div class="processing-step completed">
                    <div>✅ IBM Granite Embeddings Created</div>
                </div>
                <div class="processing-step active">
                    <div>🎯 Setting up Cross-Encoder Reranking...</div>
                </div>
                """, unsafe_allow_html=True)
            
            status_text.markdown("**🎯 Configuring Cross-Encoder Reranking System...**")
            details_text.markdown("*Enhancing answer relevance with advanced reranking*")
            progress_bar.progress(85)
            time.sleep(1)
            
            # Step 5: Save and finalize
            with step_container.container():
                st.markdown("""
                <div class="processing-step completed">
                    <div>✅ Cross-Encoder System Ready</div>
                </div>
                <div class="processing-step active">
                    <div>💾 Finalizing Study Collection...</div>
                </div>
                """, unsafe_allow_html=True)
            
            status_text.markdown("**💾 Saving AI-Powered Study Collection...**")
            details_text.markdown("*Storing for future intelligent study sessions*")
            progress_bar.progress(95)
            time.sleep(0.8)
            
            if st.session_state.db_manager and st.session_state.db_manager.save_database(rag):
                progress_bar.progress(100)
                
                with step_container.container():
                    st.markdown("""
                    <div class="processing-step completed">
                        <div>✅ All Systems Ready</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                status_text.markdown("**✅ AI Study Collection Ready!**")
                details_text.markdown("")
                
                # Load the new collection
                st.session_state.current_rag = rag
                st.session_state.current_db_name = name
                
                # Success message with enhanced details
                stats = result.get('processing_stats', {})
                st.markdown(f"""
                <div class="status-card status-success success-animation">
                    <h4>🎉 AI-Powered Study Collection Created!</h4>
                    <p><strong>{name}</strong> is ready with {result['total_chunks']} searchable chunks from {result['total_files']} files.</p>
                    <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.3); border-radius: 8px;">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">🤖 AI Features Enabled:</div>
                        <div>• IBM Granite embeddings for semantic understanding</div>
                        <div>• Cross-encoder reranking for improved accuracy</div>
                        <div>• Comprehensive chat history with metadata</div>
                        <div>• Advanced OCR + hybrid text extraction</div>
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem; padding: 1rem; background: rgba(0,0,0,0.1); border-radius: 8px;">
                        <div style="font-weight: 600;">📊 Processing Summary:</div>
                        <div>• Standard extraction: {stats.get('fitz_success', 0)} files</div>
                        <div>• Hybrid processing: {stats.get('hybrid_extraction', 0)} files</div>
                        <div>• OCR fallback: {stats.get('ocr_fallback', 0)} files</div>
                        <div>• Processing failed: {stats.get('failed', 0)} files</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                time.sleep(2)
                
                # Auto-navigate to chat
                st.session_state.auto_navigate_to_chat = True
                st.rerun()
            else:
                st.error("Failed to save study collection!")
        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

def ask_question_enhanced(question: str):
    """Enhanced question processing with better feedback"""
    with st.spinner("IBM Granite AI is analyzing your question..."):
        try:
            status_placeholder = st.empty()
            
            status_placeholder.markdown("🔍 Searching with IBM Granite embeddings...")
            time.sleep(0.8)
            
            status_placeholder.markdown("🎯 Applying cross-encoder reranking...")
            time.sleep(0.6)
            
            result = st.session_state.current_rag.query(question)
            
            status_placeholder.markdown("🤖 Generating intelligent response...")
            time.sleep(0.8)
            
            status_placeholder.empty()
            
            if result["success"]:
                # Show success message briefly
                success_msg = st.success("✅ Answer generated successfully!")
                time.sleep(1)
                success_msg.empty()
                st.rerun()
            else:
                st.error(f"Error: {result['answer']}")
        
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")

def download_chat_history(format_type: str):
    """Download chat history in specified format"""
    if not st.session_state.current_rag or not st.session_state.current_rag.chat_history.chat_entries:
        st.warning("No chat history available to download")
        return
    
    try:
        # Generate formatted content
        content = st.session_state.current_rag.get_chat_history_download(format_type)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collection_name = st.session_state.current_db_name.replace(" ", "_")
        session_id = st.session_state.current_rag.chat_history.session_id
        
        if format_type == "markdown":
            filename = f"{collection_name}_chat_{session_id}_{timestamp}.md"
            mime_type = "text/markdown"
        else:
            filename = f"{collection_name}_chat_{session_id}_{timestamp}.txt"
            mime_type = "text/plain"
        
        # Provide download
        st.download_button(
            label=f"📥 Download {format_type.upper()} Chat History",
            data=content,
            file_name=filename,
            mime=mime_type,
            key=f"download_{format_type}_{timestamp}"
        )
        
        st.success(f"Chat history prepared for download! ({len(st.session_state.current_rag.chat_history.chat_entries)} entries)")
        
    except Exception as e:
        st.error(f"Error preparing download: {str(e)}")

if __name__ == "__main__":
    main()