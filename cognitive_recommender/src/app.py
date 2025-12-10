"""
============================================================
BRAIN SPARKS - Cognitive Educational Recommender
============================================================
A complete Streamlit web application implementing the four pillars
of cognitive computing: UNDERSTAND, REASON, LEARN, INTERACT.

This application provides:
- Home Page: Welcome and search interface
- Results Page: Personalized learning paths
- Feedback Page: User feedback collection (LEARN pillar)
- About Page: System information and ethics
- Interactive Knowledge Graph visualization

Author: Rugogamu Noela
Institution: Uganda Christian University (UCU)
============================================================
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
import pickle

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

# Import our cognitive modules
try:
    from nlp_utils import QueryParser, generate_topic_explanation, generate_uganda_relevance
    from kg_utils import build_knowledge_graph_from_data
    from recommender import CognitiveRecommender
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Module import error: {e}")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Sparks - Educational Recommender",
    page_icon=":book:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - Red, White, and Blue Theme
# ============================================================
st.markdown("""
<style>

/* Hide the Streamlit top toolbar */
[data-testid="stToolbar"] {
    visibility: hidden !important;
}

/* Remove the dark theme background/header */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Remove shadow/border from header */
header[data-testid="stHeader"]::before {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-red: #C41E3A;
        --primary-red-dark: #9A1830;
        --primary-red-light: #E8456B;
        --accent-blue: #2563EB;
        --accent-blue-light: #3B82F6;
        --accent-green: #16A34A;
        --white: #FFFFFF;
        --off-white: #F8F9FA;
        --gray-100: #F1F5F9;
        --gray-200: #E2E8F0;
        --gray-500: #374151;
        --gray-700: #334155;
        --gray-900: #0F172A;
    }
    
    .stApp {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F5 30%, #F8FAFF 100%);
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-red-dark) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 50px rgba(196, 30, 58, 0.35);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 80%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: white;
        margin: 0 0 0.5rem 0;
        position: relative;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header .subtitle {
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 400;
        position: relative;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border: 2px solid var(--gray-100);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(196, 30, 58, 0.15);
        border-color: var(--primary-red);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        color: var(--gray-900);
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-family: 'Source Sans Pro', sans-serif;
        color: #334155 !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Search Container */
    .search-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border: 2px solid var(--gray-100);
    }
    
    /* Result Cards */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border-left: 5px solid var(--primary-red);
    }
    
    .result-card.blue {
        border-left-color: var(--accent-blue);
    }
    
    .result-card.green {
        border-left-color: var(--accent-green);
    }
    
    /* Learning Path Steps */
    .learning-step {
        background: linear-gradient(135deg, #FFFFFF 0%, var(--off-white) 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 2px solid var(--gray-200);
        position: relative;
        transition: all 0.3s ease;
    }
    
    .learning-step:hover {
        box-shadow: 0 15px 40px rgba(196, 30, 58, 0.12);
        transform: translateY(-3px);
        border-color: var(--primary-red);
    }
    
    .step-number {
        position: absolute;
        top: -15px;
        left: 25px;
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-red-dark) 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 5px 15px rgba(196, 30, 58, 0.4);
    }
    
    .step-title {
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        color: var(--primary-red);
        margin: 0.5rem 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .badge-article { background: #DBEAFE; color: #1E40AF; }
    .badge-video { background: #FEE2E2; color: #991B1B; }
    .badge-quiz { background: #D1FAE5; color: #065F46; }
    .badge-beginner { background: #D1FAE5; color: #065F46; }
    .badge-intermediate { background: #FEF3C7; color: #92400E; }
    .badge-advanced { background: #FEE2E2; color: #991B1B; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF5F5 100%);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid var(--gray-100);
        margin-bottom: 1.5rem;
    }
    
    .sidebar-logo {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        color: var(--primary-red);
        font-weight: 700;
    }
    
    /* Status Indicators */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background: var(--accent-green); }
    .status-warning { background: #EAB308; }
    .status-error { background: var(--primary-red); }
    
    /* Understanding Box */
    .understanding-box {
        background: linear-gradient(135deg, #FEF2F2 0%, #FFF5F5 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #FECACA;
    }
    
    /* Uganda Box */
    .uganda-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #F0FDF4 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #A7F3D0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--primary-red-dark) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-family: 'Source Sans Pro', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(196, 30, 58, 0.3);
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis;
        min-width: fit-content;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(196, 30, 58, 0.4);
    }
    
    /* Navigation buttons - prevent text wrapping */
    section[data-testid="stSidebar"] .stButton > button {
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow: visible !important;
        text-overflow: clip !important;
        min-width: 80px !important;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* GLOBAL TEXT VISIBILITY FIX */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #1a1a1a !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #1a1a1a !important;
    }
    
    /* Input labels */
    .stTextInput label, .stTextArea label, .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Radio and checkbox text */
    .stRadio label, .stCheckbox label {
        color: #1a1a1a !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    /* Metric text */
    [data-testid="stMetricValue"] {
        color: #C41E3A !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
    }
    
    /* Tab text */
    .stTabs [data-baseweb="tab"] {
        color: #1a1a1a !important;
    }
    
    /* Feedback Section */
    .feedback-box {
        background: linear-gradient(135deg, #F8FAFF 0%, #F1F5F9 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        border: 2px solid var(--gray-200);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: 2px solid var(--gray-200);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-red) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    
    if 'searched_topic' not in st.session_state:
        st.session_state.searched_topic = None
    
    if 'recommender' not in st.session_state:
        try:
            data_path = os.path.join(current_dir, '..', 'data', 'educational_content.json')
            feedback_path = os.path.join(current_dir, '..', 'data', 'feedback.json')
            st.session_state.recommender = CognitiveRecommender(data_path, feedback_path)
            st.session_state.kg = build_knowledge_graph_from_data(data_path)
        except Exception as e:
            st.session_state.recommender = None
            st.session_state.kg = None
            st.error(f"Error initializing system: {e}")

# ============================================================
# INTERACTIVE KNOWLEDGE GRAPH
# ============================================================

def create_interactive_knowledge_graph(kg, highlight_topic=None):
    """Create an interactive Plotly visualization of the knowledge graph.
    
    When a topic is selected, FILTERS the graph to show only that topic
    and its directly connected nodes (resources and applications).
    """
    
    # Get graph data
    G = kg.graph
    
    # FILTER: If a topic is selected, create a subgraph with only connected nodes
    if highlight_topic and highlight_topic in G.nodes():
        # Get all nodes connected to the selected topic
        connected_nodes = set([highlight_topic])
        
        # Add all neighbors (directly connected nodes)
        for neighbor in G.neighbors(highlight_topic):
            connected_nodes.add(neighbor)
            # Also add nodes connected to those neighbors (2nd level)
            for second_neighbor in G.neighbors(neighbor):
                connected_nodes.add(second_neighbor)
        
        # Create filtered subgraph
        G_filtered = G.subgraph(connected_nodes).copy()
    else:
        # Show full graph when no filter
        G_filtered = G
    
    # Create positions using spring layout with more spacing for clarity
    pos = nx.spring_layout(G_filtered, k=3, iterations=100, seed=42)
    
    # Separate nodes by type
    topic_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n].get('node_type') == 'topic']
    resource_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n].get('node_type') == 'resource']
    app_nodes = [n for n in G_filtered.nodes() if G_filtered.nodes[n].get('node_type') == 'application']
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G_filtered.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#94A3B8'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces for each type
    traces = [edge_trace]
    
    # Topics (red, highlighted topic in gold)
    if topic_nodes:
        topic_x = [pos[n][0] for n in topic_nodes]
        topic_y = [pos[n][1] for n in topic_nodes]
        topic_text = [G_filtered.nodes[n].get('name', n).replace('_', ' ').title() for n in topic_nodes]
        # Highlight selected topic with gold, others with red
        topic_colors = ['#FFD700' if n == highlight_topic else '#C41E3A' for n in topic_nodes]
        topic_sizes = [35 if n == highlight_topic else 25 for n in topic_nodes]
        
        traces.append(go.Scatter(
            x=topic_x, y=topic_y,
            mode='markers+text',
            hoverinfo='text',
            text=topic_text,
            textposition="top center",
            textfont=dict(size=11, color='#1E293B', family='Source Sans Pro'),
            marker=dict(
                size=topic_sizes,
                color=topic_colors,
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            name='Topics'
        ))
    
    # Resources (blue) - Show resource titles in filtered view
    if resource_nodes:
        res_x = [pos[n][0] for n in resource_nodes]
        res_y = [pos[n][1] for n in resource_nodes]
        # Show more detail when filtered
        if highlight_topic:
            res_text = [G_filtered.nodes[n].get('title', n)[:25] + '...' if len(G_filtered.nodes[n].get('title', n)) > 25 else G_filtered.nodes[n].get('title', n) for n in resource_nodes]
            res_hover = [f"{G_filtered.nodes[n].get('title', n)}\nType: {G_filtered.nodes[n].get('resource_type', 'article')}\nDifficulty: {G_filtered.nodes[n].get('difficulty', 'beginner')}" for n in resource_nodes]
        else:
            res_text = ['' for n in resource_nodes]
            res_hover = [n for n in resource_nodes]
        
        traces.append(go.Scatter(
            x=res_x, y=res_y,
            mode='markers+text' if highlight_topic else 'markers',
            hoverinfo='text',
            hovertext=res_hover,
            text=res_text,
            textposition="bottom center",
            textfont=dict(size=8, color='#1E40AF'),
            marker=dict(
                size=18 if highlight_topic else 12,
                color='#2563EB',
                line=dict(width=2, color='white'),
                symbol='square'
            ),
            name='Resources'
        ))
    
    # Applications (green) - Uganda applications
    if app_nodes:
        app_x = [pos[n][0] for n in app_nodes]
        app_y = [pos[n][1] for n in app_nodes]
        app_text = [G_filtered.nodes[n].get('name', n).replace('_', ' ').title() for n in app_nodes]
        
        traces.append(go.Scatter(
            x=app_x, y=app_y,
            mode='markers+text',
            hoverinfo='text',
            text=app_text,
            textposition="bottom center",
            textfont=dict(size=9, color='#065F46'),
            marker=dict(
                size=22 if highlight_topic else 18,
                color='#16A34A',
                line=dict(width=2, color='white'),
                symbol='diamond'
            ),
            name='Uganda Applications'
        ))
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Dynamic title based on filter
    if highlight_topic:
        title_text = f' Knowledge Graph: {highlight_topic.replace("_", " ").title()}'
        subtitle = f'Showing {len(G_filtered.nodes())} nodes connected to this topic'
    else:
        title_text = ' Full Knowledge Graph'
        subtitle = f'All {len(G_filtered.nodes())} nodes • Select a topic to filter'
    
    fig.update_layout(
        title=dict(
            text=f'{title_text}<br><sup style="color: #64748B;">{subtitle}</sup>',
            font=dict(size=18, color='#1E293B', family='Source Sans Pro')
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(248,250,252,0.8)',
        paper_bgcolor='white',
        height=550
    )
    
    return fig

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Render the sidebar with navigation and system status."""
    with st.sidebar:
        # Header
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">BS</div>
            <div class="sidebar-title">Brain Sparks</div>
            <div style="color: #374151; font-size: 0.85rem; font-weight: 500;">Cognitive Educational Recommender</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("Navigation")
        
        # Use single column layout to prevent text wrapping
        if st.button("Home", use_container_width=True, key="nav_home"):
            st.session_state.page = 'home'
            st.rerun()
        if st.button("About", use_container_width=True, key="nav_about"):
            st.session_state.page = 'about'
            st.rerun()
        if st.button("Graph", use_container_width=True, key="nav_graph"):
            st.session_state.page = 'graph'
            st.rerun()
        if st.button("Stats", use_container_width=True, key="nav_stats"):
            st.session_state.page = 'stats'
            st.rerun()
        
        st.divider()
        
        # System Status
        st.markdown("Model Status")
        
        if st.session_state.recommender:
            stats = st.session_state.recommender.get_system_stats()
            
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                <div style="margin-bottom: 0.5rem;">
                    <span class="status-dot status-active"></span>
                    <strong>NLP:</strong> {stats.get('nlp_status', 'Unknown')}
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span class="status-dot status-active"></span>
                    <strong>Knowledge Graph:</strong> {stats.get('kg_status', 'Unknown')}
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <span class="status-dot status-active"></span>
                    <strong>Recommender:</strong> {stats.get('recommender_status', 'Unknown')}
                </div>
                <div>
                    <span class="status-dot status-active"></span>
                    <strong>Feedback:</strong> {stats.get('feedback_status', 'Unknown')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("Dataset")
            st.metric("Resources", stats.get('total_resources', 0))
            st.metric("Topics", stats.get('total_topics', 0))
            st.metric("Feedback Entries", stats.get('feedback_count', 0))
        else:
            st.warning("System not initialized")
        
        st.divider()
        
        # Footer info
        st.markdown("""
        <div style="text-align: center; color: #374151; font-size: 0.8rem;">
            <strong>Last Updated:</strong> Today<br>
            <strong>User:</strong> Guest<br><br>
            Built by <strong>Rugogamu Noela</strong><br>
            UCU • 2025
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================

def render_home_page():
    """Render the home/landing page."""
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>Brain Sparks</h1>
        <div class="subtitle">
            An intelligent learning companion powered by cognitive computing.<br>
            Designed for Ugandan learners. Built by Rugogamu Noela.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div style="text-align: center; max-width: 800px; margin: 0 auto 2rem auto;">
        <p style="font-size: 1.15rem; color: #334155; line-height: 1.8;">
            Type any topic you want to learn. <strong style="color: #C41E3A;">Brain Sparks</strong> will 
            <strong>understand</strong> your query, <strong>reason</strong> over curated content, 
            <strong>learn</strong> from your feedback, and recommend a 
            <strong>personalized learning path</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("📚", "55+ Resources", "Curated educational content"),
        ("🌍", "Uganda Focus", "Local context & applications"),
        ("🧠", "AI-Powered", "Cognitive reasoning engine"),
        ("📈", "Adaptive", "Learns from your feedback")
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Search Section
    st.markdown("""
    <div class="search-container">
        <h3 style="text-align: center; color: #1E293B; margin-bottom: 1.5rem;">
             What would you like to learn today?
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Query Input
    query = st.text_area(
        "Enter your query",
        placeholder="Example: Explain the basics of quantum computing and show me how it could be relevant for solving problems in Uganda.",
        height=120,
        key="query_input",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate Learning Path →", use_container_width=True, type="primary"):
            if query and st.session_state.recommender:
                st.session_state.current_query = query
                with st.spinner(" Processing your query..."):
                    st.session_state.query_result = st.session_state.recommender.process_query(query)
                    # Store the searched topic for graph coordination
                    if st.session_state.query_result and 'parsed_query' in st.session_state.query_result:
                        st.session_state.searched_topic = st.session_state.query_result['parsed_query'].get('primary_topic')
                st.session_state.page = 'results'
                st.session_state.feedback_submitted = False
                st.rerun()
            elif not query:
                st.warning("Please enter a learning query first!")
            else:
                st.error("System not initialized. Please refresh the page.")
    
    # Sample Queries
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("Try These Sample Queries")
    
    samples = [
        "Explain the basics of quantum computing and how it could help Uganda",
        "I want to learn machine learning for agriculture",
        "How can cybersecurity protect mobile money users in Uganda?",
        "Teach me about blockchain applications in East Africa",
        "What is artificial intelligence and how can it help healthcare?"
    ]
    
    cols = st.columns(2)
    for i, sample in enumerate(samples):
        with cols[i % 2]:
            if st.button(f" {sample[:50]}...", key=f"sample_{i}", use_container_width=True):
                st.session_state.current_query = sample
                if st.session_state.recommender:
                    with st.spinner(" Processing..."):
                        st.session_state.query_result = st.session_state.recommender.process_query(sample)
                        # Store the searched topic for graph coordination
                        if st.session_state.query_result and 'parsed_query' in st.session_state.query_result:
                            st.session_state.searched_topic = st.session_state.query_result['parsed_query'].get('primary_topic')
                    st.session_state.page = 'results'
                    st.rerun()

# ============================================================
# RESULTS PAGE
# ============================================================

def render_results_page():
    """Render the results page with recommendations."""
    
    result = st.session_state.query_result
    
    if not result:
        st.warning("No results to display. Please make a query first.")
        if st.button("← Go to Home"):
            st.session_state.page = 'home'
            st.rerun()
        return
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #C41E3A 0%, #9A1830 100%); 
                padding: 1.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
                box-shadow: 0 15px 50px rgba(196, 30, 58, 0.35);">
        <h2 style="font-family: 'Playfair Display', serif; color: white; margin: 0; font-size: 1.8rem;">
             Your Personalized Learning Path
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Understanding", "Explanation", "Learning Path", "Feedback"])
    
    # TAB 1: Understanding Section
    with tab1:
        st.markdown("What You Asked For")
        
        st.markdown(f"""
        <div class="understanding-box">
            <div style="font-style: italic; color: #334155; font-size: 1.1rem; padding: 0.5rem;">
                "{st.session_state.current_query}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        parsed = result.get('parsed_query', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            topic = parsed.get('primary_topic', 'general').replace('_', ' ').title()
            confidence = parsed.get('topic_confidence', 0) * 100
            st.markdown(f"""
            <div class="result-card" style="text-align: center;">
                <div style="color: #C41E3A; font-weight: 600; text-transform: uppercase; font-size: 0.85rem;">
                    Extracted Topic
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E293B; margin: 0.5rem 0;">
                    {topic}
                </div>
                <div style="color: #374151;">Confidence: {confidence:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            uganda_ctx = parsed.get('uganda_context', {})
            has_uganda = "Yes" if uganda_ctx.get('has_uganda_context') else "No"
            categories = ', '.join(uganda_ctx.get('categories', [])[:3]) or "None detected"
            st.markdown(f"""
            <div class="result-card blue" style="text-align: center;">
                <div style="color: #2563EB; font-weight: 600; text-transform: uppercase; font-size: 0.85rem;">
                    Uganda Context
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E293B; margin: 0.5rem 0;">
                    {has_uganda}
                </div>
                <div style="color: #374151;">{categories}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            intent_dict = parsed.get('intent', {})
            intent = intent_dict.get('primary', 'learn') if isinstance(intent_dict, dict) else str(intent_dict).title() if intent_dict else 'Learn'
            intent_confidence = intent_dict.get('confidence', 0) * 100 if isinstance(intent_dict, dict) else 0
            st.markdown(f"""
            <div class="result-card green" style="text-align: center;">
                <div style="color: #16A34A; font-weight: 600; text-transform: uppercase; font-size: 0.85rem;">
                    Your Intent
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E293B; margin: 0.5rem 0;">
                    {intent.title()}
                </div>
                <div style="color: #374151;">Confidence: {intent_confidence:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 2: Explanation Section
    with tab2:
        st.markdown("Topic Summary")
        
        explanation = result.get('topic_explanation', 'No explanation available.')
        st.markdown(f"""
        <div class="result-card">
            {explanation}
        </div>
        """, unsafe_allow_html=True)
        
        # Uganda Relevance
        uganda_relevance = result.get('uganda_relevance')
        if uganda_relevance:
            st.markdown("Uganda Relevance")
            st.markdown(f"""
            <div class="uganda-box">
                {uganda_relevance}
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 3: Learning Path
    with tab3:
        st.markdown("Recommended Learning Path")
        st.markdown("*Follow these steps for the best learning experience:*")
        
        learning_path = result.get('learning_path', [])
        
        if not learning_path:
            st.info("No specific learning path could be generated. Try a more specific topic!")
        else:
            for step in learning_path:
                resource = step.get('resource', {})
                step_num = step.get('step', 1)
                
                # Get badge classes
                res_type = resource.get('type', 'article')
                difficulty = resource.get('difficulty', 'beginner')
                resource_title = resource.get('title', 'Resource')
                resource_desc = resource.get('description', 'No description available.')
                
                # Create a clean card using Streamlit columns and containers
                with st.container():
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #C41E3A 0%, #9A1830 100%); 
                                    color: white; width: 50px; height: 50px; border-radius: 50%; 
                                    display: flex; align-items: center; justify-content: center; 
                                    font-weight: 700; font-size: 1.5rem; margin-top: 0.5rem;">
                            {step_num}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"#### {step.get('title', 'Learning Step')}")
                        st.caption(step.get('subtitle', ''))
                        
                        # Resource title and info
                        st.markdown(f"**{resource_title}**")
                        
                        # Badges
                        col_badge1, col_badge2, col_badge3 = st.columns(3)
                        with col_badge1:
                            st.markdown(f'<span class="badge badge-{res_type}">{res_type.upper()}</span>', unsafe_allow_html=True)
                        with col_badge2:
                            st.markdown(f'<span class="badge badge-{difficulty}">{difficulty.upper()}</span>', unsafe_allow_html=True)
                        with col_badge3:
                            st.markdown(f'<span class="badge" style="background: #F1F5F9; color: #334155;">Duration: {resource.get("duration_minutes", 15)} min</span>', unsafe_allow_html=True)
                        
                        # Description
                        st.markdown(f"*{resource_desc[:300]}{'...' if len(resource_desc) > 300 else ''}*")
                        
                        # Why this resource
                        with st.expander("Why this resource?"):
                            st.write(step.get('reason', 'Recommended for your learning journey.'))
                        
                        # Resource link/embed
                        resource_url = resource.get('url', '')
                        if res_type == 'video' and resource_url:
                            # Check if it's a YouTube URL
                            if 'youtube.com' in resource_url or 'youtu.be' in resource_url:
                                # Extract video ID
                                video_id = None
                                if 'youtube.com/watch?v=' in resource_url:
                                    video_id = resource_url.split('watch?v=')[1].split('&')[0]
                                elif 'youtu.be/' in resource_url:
                                    video_id = resource_url.split('youtu.be/')[1].split('?')[0]
                                
                                if video_id:
                                    st.video(f"https://www.youtube.com/watch?v={video_id}")
                                else:
                                    st.markdown(f'[Watch Video]({resource_url})', unsafe_allow_html=True)
                            else:
                                # Regular video link
                                st.markdown(f'[Watch Video]({resource_url})')
                        elif res_type == 'article' and resource_url:
                            # Use a working placeholder or actual URL
                            if 'http' in resource_url:
                                st.markdown(f'[Read Full Article →]({resource_url})')
                            else:
                                st.info(f"Article: {resource_title} - Source: {resource.get('source', 'Educational Resource')}")
                        
                        # Source info
                        st.caption(f"Source: {resource.get('source', 'Unknown')}")
                        
                        st.divider()
                
                # Quiz handling
                if res_type == 'quiz' and 'questions' in resource:
                    with st.expander("Take the Quiz"):
                        for i, q in enumerate(resource.get('questions', [])):
                            st.markdown(f"**Q{i+1}: {q.get('q', '')}**")
                            answer = st.radio(
                                "Select your answer:",
                                q.get('options', []),
                                key=f"quiz_{resource.get('id')}_{i}",
                                label_visibility="collapsed"
                            )
                            if st.button(f"Check Answer", key=f"check_{resource.get('id')}_{i}"):
                                if answer == q.get('answer'):
                                    st.success("Correct!")
                                else:
                                    st.error(f"The correct answer is: {q.get('answer')}")
    
    # TAB 4: Feedback
    with tab4:
        render_feedback_section(learning_path)

# ============================================================
# FEEDBACK SECTION
# ============================================================

def render_feedback_section(learning_path):
    """Render the feedback collection section."""
    
    st.markdown("Help Brain Sparks Learn Better")
    
    if st.session_state.feedback_submitted:
        st.success("Thank you! Your feedback helps the AI learn and improve future recommendations.")
        return
    
    st.markdown("""
    <div class="feedback-box">
        <p style="color: #334155;">
            Your feedback is crucial for improving recommendations. Rate this learning path and help us serve you better!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rating = st.slider(
            "⭐ Rate this learning path (1-5 stars)",
            min_value=1,
            max_value=5,
            value=4,
            help="How relevant and helpful were these recommendations?"
        )
        
        st.markdown("**Your Rating:**")
        st.markdown("⭐" * rating + "☆" * (5-rating))
    
    with col2:
        helpful = st.radio(
            "Was this recommendation helpful?",
            options=["Yes, very helpful!", "Somewhat helpful", "Not really helpful"],
            help="This helps us understand what works"
        )
    
    comment = st.text_area(
        "Any suggestions for improvement? (Optional)",
        placeholder="Tell us what we could do better...",
        height=100
    )
    
    if st.button("Submit Feedback", type="primary"):
        if st.session_state.recommender:
            resource_id = None
            if learning_path:
                resource_id = learning_path[0].get('resource', {}).get('id')
            
            st.session_state.recommender.add_feedback(
                resource_id=resource_id or "general",
                rating=rating,
                helpful=(helpful == "Yes, very helpful!"),
                comment=comment if comment else None,
                query=st.session_state.current_query
            )
            
            st.session_state.feedback_submitted = True
            st.rerun()

# ============================================================
# KNOWLEDGE GRAPH PAGE
# ============================================================

def render_graph_page():
    """Render the interactive knowledge graph page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>Knowledge Graph Explorer</h1>
        <div class="subtitle">
            Explore the connections between topics, resources, and Uganda applications
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.session_state.kg:
        # Check if there's a searched topic from home page
        auto_selected_topic = None
        if st.session_state.searched_topic and st.session_state.searched_topic in st.session_state.kg.topics:
            auto_selected_topic = st.session_state.searched_topic
            st.success(f" Showing graph for your searched topic: **{auto_selected_topic.replace('_', ' ').title()}**")
        
        # Instructions
        if auto_selected_topic:
            st.info(f" This graph is automatically filtered for '{auto_selected_topic.replace('_', ' ').title()}' based on your search. You can change the topic below or view the full graph.")
        else:
            st.info("Select a topic below to filter the graph and see only related resources and applications. Choose 'Show All Topics' to see the complete knowledge graph.")
        
        # Topic selector with better labeling
        topics = list(st.session_state.kg.topics)
        topic_options = ['Show All Topics (Full Graph)'] + sorted([t.replace('_', ' ').title() for t in topics])
        
        # Set default selection to searched topic if available
        default_index = 0
        if auto_selected_topic:
            topic_display = auto_selected_topic.replace('_', ' ').title()
            if topic_display in topic_options:
                default_index = topic_options.index(topic_display)
        
        selected_display = st.selectbox(
            "Filter by Topic:",
            options=topic_options,
            index=default_index,
            help="Select a specific topic to narrow down the graph to only related nodes"
        )
        
        # Convert display name back to internal name
        if selected_display == 'Show All Topics (Full Graph)':
            highlight = None
        else:
            highlight = selected_display.lower().replace(' ', '_')
        
        # Visualization method selector
        viz_method = st.radio(
            "Visualization Method:",
            ["PyVis (Interactive HTML)", "Plotly (Embedded)"],
            horizontal=True,
            help="PyVis provides more interactive features, Plotly is embedded in the page"
        )
        
        # Create and display graph
        if viz_method == "PyVis (Interactive HTML)":
            try:
                from kg_utils import create_pyvis_graph
                import tempfile
                import os
                
                # Create temporary HTML file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html', dir=current_dir) as tmp_file:
                    tmp_path = tmp_file.name
                
                # Generate PyVis graph
                graph_path = create_pyvis_graph(st.session_state.kg, topic=highlight, output_path=tmp_path)
                
                # Read and display the HTML
                with open(graph_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                components.html(html_content, height=800, scrolling=True)
                
                # Clean up
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
            except ImportError:
                st.warning("PyVis not available. Install with: pip install pyvis")
                st.info("Falling back to Plotly visualization...")
                fig = create_interactive_knowledge_graph(st.session_state.kg, highlight)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating PyVis graph: {e}")
                st.info("Falling back to Plotly visualization...")
                fig = create_interactive_knowledge_graph(st.session_state.kg, highlight)
                st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_interactive_knowledge_graph(st.session_state.kg, highlight)
            st.plotly_chart(fig, use_container_width=True)
        
        # Graph statistics
        st.markdown("Graph Statistics")
        
        stats = st.session_state.kg.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Nodes", stats.get('total_nodes', 0))
        col2.metric("Total Edges", stats.get('total_edges', 0))
        col3.metric("Topics", stats.get('topics', 0))
        col4.metric("Resources", stats.get('resources', 0))
        
        # Show topic details if selected
        if highlight:
            st.markdown(f"Resources for '{highlight.replace('_', ' ').title()}'")
            st.markdown("*These are the learning resources connected to this topic:*")
            
            resources = st.session_state.kg.get_resources_for_topic(highlight)
            
            if resources:
                for r in resources[:8]:  # Show more resources
                    res_type = r.get('resource_type', 'article')
                    difficulty = r.get('difficulty', 'beginner')
                    st.markdown(f"""
                    <div class="result-card" style="margin-bottom: 0.5rem;">
                        <strong style="color: #1E293B;">{r.get('title', 'Unknown')}</strong><br>
                        <span class="badge badge-{res_type}">{res_type.upper()}</span>
                        <span class="badge badge-{difficulty}">{difficulty.upper()}</span>
                        <p style="color: #374151; font-size: 0.9rem; margin-top: 0.5rem;">{r.get('description', '')[:150]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No resources found for this topic.")
    else:
        st.error("Knowledge graph not initialized.")

# ============================================================
# STATS PAGE
# ============================================================

def render_stats_page():
    """Render the statistics and analytics page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>System Analytics</h1>
        <div class="subtitle">
            Performance metrics and dataset statistics
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    if st.session_state.recommender:
        stats = st.session_state.recommender.get_system_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Resources", stats.get('total_resources', 0))
        col2.metric("Topics Covered", stats.get('total_topics', 0))
        col3.metric("Applications", stats.get('total_applications', 0))
        col4.metric("Feedback Entries", stats.get('feedback_count', 0))
        
        # Load data for visualizations
        data_path = os.path.join(current_dir, '..', 'data', 'educational_content.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data['resources'])
        
        st.markdown("Dataset Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Type distribution
            fig_type = px.pie(
                df, names='type', 
                title='Resource Types',
                color_discrete_sequence=['#C41E3A', '#2563EB', '#16A34A']
            )
            fig_type.update_layout(height=350)
            st.plotly_chart(fig_type, use_container_width=True)
        
        with col2:
            # Difficulty distribution
            fig_diff = px.bar(
                df['difficulty'].value_counts().reset_index(),
                x='difficulty', y='count',
                title='Difficulty Levels',
                color='difficulty',
                color_discrete_map={
                    'beginner': '#16A34A',
                    'intermediate': '#EAB308',
                    'advanced': '#C41E3A'
                }
            )
            fig_diff.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_diff, use_container_width=True)
        
        # Topic distribution
        st.markdown("Topics Distribution")
        topic_counts = df['topic'].value_counts().head(15)
        
        fig_topics = px.bar(
            x=topic_counts.values,
            y=[t.replace('_', ' ').title() for t in topic_counts.index],
            orientation='h',
            title='Top 15 Topics',
            color=topic_counts.values,
            color_continuous_scale='Reds'
        )
        fig_topics.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_topics, use_container_width=True)
    else:
        st.error("System not initialized.")

# ============================================================
# ABOUT PAGE
# ============================================================

def render_about_page():
    """Render the about page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>About Brain Sparks</h1>
        <div class="subtitle">
            Understanding the Cognitive Computing Approach
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    # What is Cognitive Computing
    st.markdown("What is Cognitive Computing?")
    st.markdown("""
    <div class="result-card">
        <p><strong>Cognitive computing</strong> refers to computer systems that simulate human thought processes. 
        Unlike traditional programs that follow rigid rules, cognitive systems can:</p>
        <ul>
            <li><strong>Understand</strong> natural language and context</li>
            <li><strong>Reason</strong> over large amounts of information</li>
            <li><strong>Learn</strong> from interactions and improve over time</li>
            <li><strong>Interact</strong> naturally with humans</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # The Four Pillars
    st.markdown("The Four Pillars")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #C41E3A;">UNDERSTAND Pillar</h4>
            <p>Uses <strong>Natural Language Processing</strong> to:</p>
            <ul>
                <li>Parse your queries</li>
                <li>Extract topics and intent</li>
                <li>Detect Uganda context</li>
            </ul>
            <p><em>Technology: NLTK, Custom NLP</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-card blue">
            <h4 style="color: #2563EB;">LEARN Pillar</h4>
            <p>Implements <strong>continuous improvement</strong>:</p>
            <ul>
                <li>Collects user feedback</li>
                <li>Adjusts recommendation scores</li>
                <li>Improves over time</li>
            </ul>
            <p><em>Technology: Feedback weighting</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #C41E3A;">REASON Pillar</h4>
            <p>Uses a <strong>Knowledge Graph</strong> to:</p>
            <ul>
                <li>Connect topics and resources</li>
                <li>Find prerequisites</li>
                <li>Generate learning paths</li>
            </ul>
            <p><em>Technology: NetworkX, TF-IDF</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-card green">
            <h4 style="color: #16A34A;">INTERACT Pillar</h4>
            <p>Provides an <strong>intuitive interface</strong>:</p>
            <ul>
                <li>Natural language input</li>
                <li>Visual learning paths</li>
                <li>Interactive quizzes</li>
            </ul>
            <p><em>Technology: Streamlit</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Ethics

    # Creator
    st.markdown("Creator")
    st.markdown("""
    <div class="result-card" style="text-align: center;">
        <h3>Rugogamu Noela</h3>
        <p>Uganda Christian University (UCU)</p>
        <p style="color: #374151;">
            Cognitive Computing Project • December 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Route to appropriate page
    page = st.session_state.page
    
    if page == 'home':
        render_home_page()
    elif page == 'results':
        render_results_page()
    elif page == 'graph':
        render_graph_page()
    elif page == 'stats':
        render_stats_page()
    elif page == 'about':
        render_about_page()
    else:
        render_home_page()
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 2rem; border-top: 2px solid #E2E8F0; margin-top: 2rem;">
        <p style="color: #374151;">
            © 2025 <strong style="color: #C41E3A;">Brain Sparks</strong> — 
            Built by <strong>Rugogamu Noela</strong>, Uganda Christian University (UCU)<br>
            Cognitive Computing Project • For Academic Use
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
