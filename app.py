"""
🗳️ Political Tweet Intensity Analyzer

A Streamlit app for analyzing political sentiment and intensity in tweets using DistilBERT.
Tracks how current political discourse compares to 2021 senator baseline.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any
import pandas as pd

# Import our custom modules with error handling for Streamlit Cloud
try:
    from political_analyzer import PoliticalAnalyzer
    from tweet_tracker import TweetTracker
    from analytics_dashboard import AnalyticsDashboard
    import pandas as pd
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Political Tweet Intensity Analyzer",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Debug info for Streamlit Cloud
if st.sidebar.button("🔍 Debug Info"):
    st.sidebar.write("**Environment Info:**")
    import sys
    st.sidebar.write(f"Python: {sys.version}")
    st.sidebar.write(f"Streamlit: {st.__version__}")
    try:
        import transformers
        st.sidebar.write(f"Transformers: {transformers.__version__}")
    except:
        st.sidebar.write("Transformers: Not available")
    try:
        import torch
        st.sidebar.write(f"PyTorch: {torch.__version__}")
    except:
        st.sidebar.write("PyTorch: Not available")

# Custom CSS for political theme
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Slab:wght@400;600;700&display=swap');

/* CSS Variables */
:root {
    --primary-blue: #1f4788;
    --primary-red: #c41e3a;
    --neutral-gray: #6c757d;
    --light-blue: #e3f2fd;
    --light-red: #ffebee;
    --dark-text: #212529;
    --light-text: #6c757d;
}

/* Main styling */
.main {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary-blue);
    margin-bottom: 1rem;
}

.intensity-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 0.5rem;
    border-left: 3px solid #dc3545;
}

/* Political direction styling */
.democratic {
    color: var(--primary-blue);
    font-weight: 600;
}

.republican {
    color: var(--primary-red);
    font-weight: 600;
}

.neutral {
    color: var(--neutral-gray);
    font-weight: 500;
}

/* Intensity indicators */
.intensity-high { border-left-color: #dc3545 !important; }
.intensity-medium { border-left-color: #ffc107 !important; }
.intensity-low { border-left-color: #28a745 !important; }

/* Analytics container */
.analytics-container {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, var(--primary-blue), var(--primary-red));
    color: white;
    border-radius: 12px;
    margin-bottom: 2rem;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary-blue), #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(31, 71, 136, 0.3);
}

/* Warning/info boxes */
.baseline-info {
    background: var(--light-blue);
    border: 1px solid var(--primary-blue);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.comparison-warning {
    background: var(--light-red);
    border: 1px solid var(--primary-red);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'tweet_tracker' not in st.session_state:
    st.session_state.tweet_tracker = TweetTracker()

if 'political_analyzer' not in st.session_state:
    try:
        with st.spinner("🤖 Loading DistilBERT model (this may take a moment on first run)..."):
            st.session_state.political_analyzer = PoliticalAnalyzer()
    except Exception as e:
        st.error(f"❌ Failed to load political analyzer: {e}")
        st.info("💡 This might be a temporary issue with model downloading. Try refreshing the page.")
        # Create a dummy analyzer to prevent crashes
        st.session_state.political_analyzer = None

if 'analytics_dashboard' not in st.session_state:
    st.session_state.analytics_dashboard = AnalyticsDashboard(st.session_state.tweet_tracker)

def main():
    # App header
    st.markdown("""
    <div class="app-header">
        <h1>🗳️ Political Tweet Intensity Analyzer</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9;">
            Analyze political sentiment and intensity in tweets using AI
        </p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.7;">
            Compare current discourse to 2021 senator baseline
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x100/1f4788/ffffff?text=🗳️", width=150)
        st.markdown("### Navigation")
        
        page = st.radio(
            "Choose a section:",
            ["🔍 Analyze Tweets", "📊 Analytics Dashboard", "📈 Historical Trends", "🔬 Model Testing", "ℹ️ About"],
            index=0
        )
        
        st.markdown("---")
        
        # Model status
        if st.session_state.political_analyzer and st.session_state.political_analyzer.model_available:
            st.success("🤖 DistilBERT Model Ready")
        elif st.session_state.political_analyzer is None:
            st.error("❌ Model Failed to Load")
            st.info("Try refreshing the page")
        else:
            st.error("❌ Model Loading Failed")
        
        # Quick stats
        try:
            summary = st.session_state.tweet_tracker.get_summary_stats()
            if summary:
                st.markdown("### 📈 Quick Stats")
                st.metric("Tweets Analyzed", summary.get('total_tweets', 0))
                st.metric("Avg Intensity", f"{summary.get('avg_intensity', 0):.2f}")
                
                if summary.get('most_extreme_direction'):
                    direction_color = "democratic" if summary['most_extreme_direction'] == 'Democratic' else "republican"
                    st.markdown(f"**Trending:** <span class='{direction_color}'>{summary['most_extreme_direction']}</span>", unsafe_allow_html=True)
        except Exception as e:
            st.sidebar.error(f"Stats error: {e}")

    # Main content based on navigation
    if page == "🔍 Analyze Tweets":
        show_analyze_page()
    elif page == "📊 Analytics Dashboard":
        show_analytics_page()
    elif page == "📈 Historical Trends":
        show_trends_page()
    elif page == "🔬 Model Testing":
        show_model_testing_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_analyze_page():
    st.title("🔍 Tweet Analysis")
    
    # Handle example selection
    if 'selected_example' in st.session_state:
        example_text = st.session_state.selected_example
        del st.session_state.selected_example
    else:
        example_text = ""
    
    # Input methods
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Tweet Text")
        
        # Text input with better handling
        tweet_text = st.text_area(
            "Paste a tweet or political statement:",
            value=example_text,  # Use the example text if selected
            height=120,
            placeholder="e.g., 'Healthcare should be a human right for all Americans'",
            help="Enter any political text to analyze. Minimum 10 characters."
        )
        
        # Character count and validation
        char_count = len(tweet_text) if tweet_text else 0
        if char_count > 0:
            if char_count < 10:
                st.warning(f"⚠️ Text too short ({char_count} chars). Need at least 10 characters.")
            elif char_count > 500:
                st.warning(f"⚠️ Text is long ({char_count} chars). Model works best with shorter text.")
            else:
                st.success(f"✅ Ready to analyze ({char_count} characters)")
        
        # Quick examples
        st.markdown("**📝 Quick Examples:**")
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            if st.button("🔵 Democratic Example", key="dem_ex"):
                st.session_state.selected_example = "Healthcare is a human right and we must ensure universal coverage for all Americans."
                st.rerun()
                
            if st.button("🔴 Republican Example", key="rep_ex"):
                st.session_state.selected_example = "We need to cut taxes and reduce government spending to unleash economic growth."
                st.rerun()
        
        with example_col2:
            if st.button("⚖️ Neutral Example", key="neutral_ex"):
                st.session_state.selected_example = "Education is important for preparing our children for the future."
                st.rerun()
                
            if st.button("🔥 Extreme Example", key="extreme_ex"):
                st.session_state.selected_example = "The radical left wants to destroy America with their socialist agenda!"
                st.rerun()
    
    with col2:
        st.markdown("### 📊 Model Info")
        
        # Show label swap status
        if st.session_state.political_analyzer and hasattr(st.session_state.political_analyzer, 'swap_labels'):
            if st.session_state.political_analyzer.swap_labels:
                st.warning("🔄 **Label Swapping ACTIVE** - Model labels are reversed")
            else:
                st.info("📋 **Normal Labels** - Using standard interpretation")
        
        st.markdown("""
        <div style="background: rgba(31, 71, 136, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <strong>🎯 Model Details:</strong><br>
            • Trained on 2021 senator tweets<br>
            • 90.76% accuracy<br>
            • Binary classification (Dem/Rep)<br>
            • Optimized for short text
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(196, 30, 58, 0.1); padding: 1rem; border-radius: 8px;">
            <strong>⚠️ Baseline Context:</strong><br>
            Results compare to 2021 senator discourse. 
            Higher intensity may indicate more extreme 
            language than typical political speech.
        </div>
        """, unsafe_allow_html=True)
        
        # Debug info
        if st.button("🔍 Open Model Tester"):
            st.info("💡 Go to the '🔬 Model Testing' page in the sidebar for comprehensive debugging tools.")

    # Analysis section - improved logic
    analysis_ready = tweet_text and len(tweet_text.strip()) >= 10
    
    if analysis_ready:
        # Big analyze button
        analyze_button = st.button(
            "🔍 Analyze Political Content", 
            type="primary",
            use_container_width=True,
            help="Click to analyze the political sentiment and intensity"
        )
        
        if analyze_button:
            # Check if model is available
            if not st.session_state.political_analyzer or not st.session_state.political_analyzer.model_available:
                st.error("❌ Model not available. Please refresh the page to retry model loading.")
                if st.button("🔄 Retry Model Loading"):
                    st.session_state.political_analyzer = None
                    st.rerun()
                return
                
            with st.spinner("🤖 Analyzing political intensity..."):
                try:
                    # Clean the input text
                    clean_text = tweet_text.strip()
                    
                    # Analyze the tweet
                    analysis_result = st.session_state.political_analyzer.analyze_tweet(clean_text)
                    
                    if analysis_result and 'error' not in analysis_result:
                        # Log the analysis
                        st.session_state.tweet_tracker.log_tweet(
                            tweet_text=clean_text,
                            analysis_result=analysis_result,
                            session_id=st.session_state.session_id
                        )
                        
                        # Display results
                        show_analysis_results(analysis_result, clean_text)
                        
                        # Check for potential misclassification and suggest label swapping
                        predicted_lean = analysis_result['political_lean']
                        confidence = analysis_result.get('confidence', 0)
                        
                        # Check for obvious keyword mismatches
                        democratic_keywords = ['healthcare', 'human right', 'universal', 'social safety', 'climate action', 'expand', 'public option', 'minimum wage']
                        republican_keywords = ['cut taxes', 'reduce spending', 'border security', 'second amendment', 'free market', 'law and order']
                        
                        has_dem_keywords = any(keyword in clean_text.lower() for keyword in democratic_keywords)
                        has_rep_keywords = any(keyword in clean_text.lower() for keyword in republican_keywords)
                        
                        # Show label swap suggestion for obvious misclassifications
                        swap_enabled = getattr(st.session_state.political_analyzer, 'swap_labels', False)
                        
                        if not swap_enabled and ((has_dem_keywords and predicted_lean == 'Republican' and confidence > 80) or \
                           (has_rep_keywords and predicted_lean == 'Democratic' and confidence > 80)):
                            
                            st.warning("🤔 **Does this classification look wrong?**")
                            expected = 'Democratic' if has_dem_keywords else 'Republican'
                            st.write(f"This appears to contain {expected.lower()} keywords but was classified as {predicted_lean}.")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔧 Try Label Swapping"):
                                    st.session_state.political_analyzer.set_label_swap(True)
                                    st.info("✅ Label swapping enabled for future analyses.")
                                    st.rerun()
                            with col2:
                                if st.button("📊 Test All Examples"):
                                    st.info("💡 Go to '🔬 Model Testing' to run comprehensive accuracy tests.")
                        
                        elif swap_enabled:
                            st.info("🔄 **Label swapping is active** - Model labels are being reversed.")
                        
                    else:
                        # Show the actual error message
                        error_msg = analysis_result.get('error', 'Unknown error occurred') if analysis_result else 'No result returned'
                        st.error(f"❌ Analysis failed: {error_msg}")
                        
                        # Show debug info
                        with st.expander("🔍 Debug Information"):
                            st.write("**Input text:**", repr(clean_text))
                            st.write("**Text length:**", len(clean_text))
                            st.write("**Model available:**", st.session_state.political_analyzer.model_available if st.session_state.political_analyzer else "No analyzer")
                            if analysis_result:
                                st.write("**Raw result:**", analysis_result)
                        
                        st.info("💡 Try a different text or use the model tester for debugging.")
                
                except Exception as e:
                    st.error(f"❌ Unexpected error during analysis: {e}")
                    with st.expander("🔍 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        # Show disabled button when not ready
        st.button(
            "🔍 Enter text above to analyze", 
            disabled=True,
            use_container_width=True,
            help="Enter at least 10 characters of political text to enable analysis"
        )
    
    # Recent analyses
    show_recent_analyses()

def show_analysis_results(analysis: Dict[str, Any], tweet_text: str):
    """Display the analysis results in a structured format"""
    
    st.markdown("### 🎯 Analysis Results")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        direction_color = "democratic" if analysis['political_lean'] == 'Democratic' else "republican"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Political Lean</h4>
            <p class="{direction_color}" style="font-size: 1.5rem; margin: 0;">
                {analysis['political_lean']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        intensity_level = "high" if analysis['intensity_scale'] > 70 else "medium" if analysis['intensity_scale'] > 40 else "low"
        st.markdown(f"""
        <div class="metric-card intensity-{intensity_level}">
            <h4>Intensity Scale</h4>
            <p style="font-size: 1.5rem; margin: 0;">
                {analysis['intensity_scale']:.1f}/100
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Confidence</h4>
            <p style="font-size: 1.5rem; margin: 0;">
                {analysis['confidence']:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Baseline Comparison</h4>
            <p style="font-size: 1rem; margin: 0;">
                {analysis['vs_baseline']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown("### 📊 Detailed Breakdown")
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        # Score breakdown
        st.markdown("**Raw Scores:**")
        st.write(f"• Democratic Score: {analysis['dem_score']:.3f}")
        st.write(f"• Republican Score: {analysis['rep_score']:.3f}")
        st.write(f"• Partisan Intensity: {analysis['partisan_intensity']:.3f}")
        st.write(f"• Extremism Score: {analysis['extremism_score']:.3f}")
    
    with detail_col2:
        # Interpretation
        st.markdown("**Interpretation:**")
        
        if analysis['intensity_scale'] > 80:
            st.error("🔥 **Very High Intensity** - Extremely partisan language")
        elif analysis['intensity_scale'] > 60:
            st.warning("⚡ **High Intensity** - Strong partisan language") 
        elif analysis['intensity_scale'] > 30:
            st.info("📊 **Moderate Intensity** - Some partisan elements")
        else:
            st.success("✅ **Low Intensity** - Relatively neutral language")
        
        if analysis['confidence'] < 60:
            st.warning("⚠️ Model has low confidence in this classification")
    
    # Visual representation
    st.markdown("### 📈 Visual Analysis")
    
    # Create intensity gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = analysis['intensity_scale'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Political Intensity"},
        delta = {'reference': 65, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Score comparison bar chart
    scores_fig = go.Figure(data=[
        go.Bar(name='Democratic', x=['Score'], y=[analysis['dem_score']], marker_color='blue'),
        go.Bar(name='Republican', x=['Score'], y=[analysis['rep_score']], marker_color='red')
    ])
    
    scores_fig.update_layout(
        title="Democratic vs Republican Scores",
        yaxis_title="Score",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(scores_fig, use_container_width=True)

def show_model_testing_page():
    """Show model testing and debugging page"""
    st.title("🔬 Model Testing & Debugging")
    st.markdown("*Comprehensive testing of the DistilBERT political classification model*")
    
    # Quick label accuracy test
    st.write("## 🚨 Quick Label Accuracy Test")
    st.markdown("""
    **Issue detected**: The model may have reversed labels where "Republican" actually means Democratic and vice versa.
    Let's test both interpretations:
    """)
    
    if st.button("🧪 Test Label Accuracy", type="primary"):
        if st.session_state.political_analyzer and st.session_state.political_analyzer.model_available:
            with st.spinner("Testing both normal and swapped label interpretations..."):
                try:
                    test_results = st.session_state.political_analyzer.test_label_accuracy()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Normal Labels Accuracy", f"{test_results['normal_accuracy']:.1f}%")
                    with col2:
                        st.metric("Swapped Labels Accuracy", f"{test_results['swapped_accuracy']:.1f}%")
                    
                    if test_results['recommendation'] == 'swap_labels':
                        st.error("🚨 **LABELS ARE REVERSED!** The model performs much better with swapped labels.")
                        
                        if st.button("🔧 Enable Label Swapping"):
                            st.session_state.political_analyzer.set_label_swap(True)
                            st.success("✅ Label swapping enabled! Try analyzing tweets again.")
                            st.info("💡 Go back to 'Analyze Tweets' and test with the examples.")
                    
                    elif test_results['normal_accuracy'] < 30:
                        st.error("🚨 **MODEL IS PERFORMING VERY POORLY** - Both interpretations have low accuracy.")
                    
                    else:
                        st.success("✅ **MODEL LABELS ARE CORRECT** - Normal interpretation works best.")
                    
                    # Show detailed results
                    with st.expander("📊 Detailed Test Results"):
                        st.write("### Normal Interpretation Results")
                        normal_df = pd.DataFrame(test_results['results']['normal']['details'])
                        if not normal_df.empty:
                            st.dataframe(normal_df)
                        
                        st.write("### Swapped Interpretation Results") 
                        swapped_df = pd.DataFrame(test_results['results']['swapped']['details'])
                        if not swapped_df.empty:
                            st.dataframe(swapped_df)
                            
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
        else:
            st.error("❌ Model not available")
    
    # Current label mode indicator
    if st.session_state.political_analyzer and st.session_state.political_analyzer.model_available:
        swap_status = getattr(st.session_state.political_analyzer, 'swap_labels', False)
        if swap_status:
            st.success("🔄 **Label Swapping ENABLED** - 'Republican' labels interpreted as Democratic")
        else:
            st.info("📋 **Normal Labels** - 'Republican' interpreted as Republican")
        
        # Manual toggle
        st.write("### Manual Label Control")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Enable Label Swapping"):
                st.session_state.political_analyzer.set_label_swap(True)
                st.success("Label swapping enabled")
                st.rerun()
        with col2:
            if st.button("📋 Use Normal Labels"):
                st.session_state.political_analyzer.set_label_swap(False)
                st.success("Normal labels enabled")
                st.rerun()
    
    st.markdown("---")
    
    # Full model tester
    try:
        from model_tester import show_model_tester
        show_model_tester()
    except ImportError:
        st.error("❌ Model tester not available")
        st.info("Make sure `model_tester.py` is in your project directory")
        
        # Basic testing functionality
        st.title("🔬 Basic Model Testing")
        
        if st.session_state.political_analyzer and st.session_state.political_analyzer.model_available:
            st.success("✅ Model is loaded and available")
            
            # Quick test
            test_text = st.text_input("Enter text to test:", "Healthcare is a human right")
            if st.button("🧪 Quick Test") and test_text:
                with st.spinner("Testing..."):
                    result = st.session_state.political_analyzer.analyze_tweet(test_text)
                    if 'error' not in result:
                        st.success(f"✅ Classification: {result['political_lean']} ({result['confidence']:.1f}% confidence)")
                        st.json(result)
                    else:
                        st.error(f"❌ Test failed: {result['error']}")
            
            # Model info
            if st.button("ℹ️ Show Raw Model Info"):
                if hasattr(st.session_state.political_analyzer, 'model'):
                    st.write("**Model Config:**")
                    config = st.session_state.political_analyzer.model.config
                    st.json(config.to_dict())
        else:
            st.error("❌ Model not available")
            if st.button("🔄 Retry Model Loading"):
                st.session_state.political_analyzer = None
                st.rerun()

def show_recent_analyses():
    """Show recent tweet analyses"""
    st.markdown("### 📚 Recent Analyses")
    
    recent_tweets = st.session_state.tweet_tracker.get_recent_tweets(limit=5)
    
    if recent_tweets:
        for i, tweet_data in enumerate(recent_tweets):
            with st.expander(f"Analysis #{len(recent_tweets)-i}: {tweet_data['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Tweet:** {tweet_data['tweet_text']}")
                st.write(f"**Political Lean:** {tweet_data['political_lean']}")
                st.write(f"**Intensity:** {tweet_data['intensity_scale']:.1f}/100")
                st.write(f"**Baseline Comparison:** {tweet_data['vs_baseline']}")
    else:
        st.info("No recent analyses. Submit a tweet above to get started!")

def show_analytics_page():
    """Show analytics dashboard"""
    st.session_state.analytics_dashboard.show_dashboard()

def show_trends_page():
    """Show historical trends analysis"""
    st.title("📈 Historical Trends")
    
    st.markdown("""
    ### Political Discourse Evolution
    
    This section compares current political language intensity to the 2021 senator baseline.
    """)
    
    # Get trend data
    trend_data = st.session_state.tweet_tracker.get_trend_analysis()
    
    if trend_data and len(trend_data) > 1:
        # Create trend visualization
        df = pd.DataFrame(trend_data)
        
        # Daily intensity trend
        fig = px.line(df, x='date', y='avg_intensity', 
                     title='Average Political Intensity Over Time',
                     labels={'avg_intensity': 'Average Intensity', 'date': 'Date'})
        
        # Add baseline reference line
        fig.add_hline(y=65, line_dash="dash", line_color="gray", 
                     annotation_text="2021 Senator Baseline")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Political direction trend
        direction_fig = px.histogram(df, x='date', color='dominant_lean',
                                   title='Political Direction Trends',
                                   color_discrete_map={'Democratic': 'blue', 'Republican': 'red'})
        
        st.plotly_chart(direction_fig, use_container_width=True)
        
    else:
        st.info("Need more data to show trends. Analyze more tweets to see patterns over time!")
    
    # Comparison with historical periods
    st.markdown("### 📊 Period Comparisons")
    
    periods = st.session_state.tweet_tracker.get_period_comparisons()
    
    if periods:
        for period_name, stats in periods.items():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{period_name} - Avg Intensity", f"{stats['avg_intensity']:.1f}")
            with col2:
                st.metric(f"{period_name} - Total Tweets", stats['total_tweets'])
            with col3:
                st.metric(f"{period_name} - Extreme Tweets", stats['extreme_count'])

def show_about_page():
    """Show about page with model and methodology info"""
    st.title("ℹ️ About This Tool")
    
    st.markdown("""
    ## 🎯 Purpose
    
    This tool analyzes political sentiment and intensity in tweets using a fine-tuned DistilBERT model. 
    It helps track how current political discourse compares to baseline political language from 2021.
    
    ## 🤖 Model Details
    
    **Model:** `m-newhauser/distilbert-political-tweets`
    - **Base Model:** DistilBERT (distilbert-base-uncased)
    - **Training Data:** 99,693 tweets from US senators during 2021
    - **Accuracy:** 90.76%
    - **Task:** Binary classification (Democratic vs Republican sentiment)
    
    ## 📊 Metrics Explained
    
    ### Political Lean
    Determines whether the text leans Democratic or Republican based on language patterns.
    
    ### Intensity Scale (0-100)
    Measures how partisan or extreme the language is:
    - **0-30:** Neutral/mild political language
    - **30-60:** Moderate partisan content  
    - **60-80:** Strong partisan language
    - **80-100:** Extremely partisan/intense language
    
    ### Baseline Comparison
    Compares intensity to 2021 senator tweets:
    - **Much more extreme:** Significantly above typical senator discourse
    - **More extreme:** Above typical senator discourse
    - **Similar:** Within normal range of political speech
    - **Less extreme:** Below typical political intensity
    
    ## ⚠️ Limitations
    
    - **Training Bias:** Model trained only on senator tweets from 2021
    - **Context Limitation:** Works best on tweet-length text
    - **Binary Classification:** Only distinguishes Dem/Rep, not independents
    - **Temporal Bias:** May not capture evolving political language
    
    ## 🔬 Methodology
    
    1. **Text Processing:** Input text tokenized using DistilBERT tokenizer
    2. **Classification:** Model outputs probability scores for each political direction
    3. **Intensity Calculation:** Based on confidence and partisan separation
    4. **Baseline Comparison:** Statistical comparison to 2021 training data distribution
    
    ## 🎓 Research Applications
    
    This tool can be used for:
    - Political discourse analysis
    - Media bias research  
    - Tracking political polarization
    - Social media content analysis
    - Educational demonstrations of NLP
    
    ## 📝 Data Privacy
    
    - Tweet text is stored locally for analysis trends
    - No personal information is collected
    - Data can be exported or cleared at any time
    - Session IDs are anonymized
    
    ## 🔧 Technical Details
    
    Built with:
    - **Frontend:** Streamlit
    - **ML Model:** Hugging Face Transformers (DistilBERT)
    - **Visualization:** Plotly
    - **Data Storage:** Local CSV files
    - **Analytics:** Pandas + custom metrics
    
    ## 📚 References
    
    - [Model on Hugging Face](https://huggingface.co/m-newhauser/distilbert-political-tweets)
    - [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
    - [Senator Tweet Dataset](https://huggingface.co/datasets/m-newhauser/senator-tweets)
    """)

if __name__ == "__main__":
    main()