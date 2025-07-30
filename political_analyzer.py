"""
🧭 Political Compass Analyzer - Streamlit App

A simple interface for analyzing political text using your fine-tuned DistilBERT model
and visualizing results on a 2D political compass.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your model (make sure distilbert_scorer.py is in the same directory)
try:
    from distilbert_scorer import DeepSeekPoliticalScorer
    MODEL_AVAILABLE = True
except ImportError as e:
    st.error(f"Could not import DeepSeekPoliticalScorer: {e}")
    st.error("Make sure distilbert_scorer.py is in your Python path")
    MODEL_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Political Compass Analyzer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'scorer' not in st.session_state:
        if MODEL_AVAILABLE:
            with st.spinner("Loading fine-tuned BERT political scorer..."):
                try:
                    st.session_state.scorer = DeepSeekPoliticalScorer("deepseek-political-scorer")
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.session_state.scorer = None
                    st.session_state.model_loaded = False
        else:
            st.session_state.scorer = None
            st.session_state.model_loaded = False
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def analyze_text(text):
    """Analyze text using your fine-tuned BERT model"""
    if not st.session_state.model_loaded or st.session_state.scorer is None:
        return {"error": "Model not available"}
    
    if not text or len(text.strip()) < 5:
        return {"error": "Text too short (minimum 5 characters)"}
    
    try:
        # Use your model's score_text method
        result = st.session_state.scorer.score_text(text.strip())
        
        # Add additional metrics
        economic_score = result['economic_score']
        social_score = result['social_score']
        
        # Calculate quadrant and intensity
        distance_from_center = np.sqrt(economic_score**2 + social_score**2)
        ideological_intensity = distance_from_center * 100
        
        # Determine quadrant
        if economic_score < 0 and social_score < 0:
            quadrant = "Liberal Economic + Liberal Social"
        elif economic_score < 0 and social_score > 0:
            quadrant = "Liberal Economic + Conservative Social"
        elif economic_score > 0 and social_score < 0:
            quadrant = "Conservative Economic + Liberal Social"
        else:
            quadrant = "Conservative Economic + Conservative Social"
        
        # Add to result
        result.update({
            'text': text,
            'quadrant': quadrant,
            'distance_from_center': distance_from_center,
            'ideological_intensity': ideological_intensity,
            'timestamp': datetime.now()
        })
        
        return result
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

def create_plotly_compass():
    """Create an interactive political compass using Plotly"""
    fig = go.Figure()
    
    # Add quadrant backgrounds
    quadrant_colors = {
        'liberal_liberal': 'rgba(33, 150, 243, 0.1)',      # Light blue
        'liberal_conservative': 'rgba(156, 39, 176, 0.1)',  # Light purple
        'conservative_liberal': 'rgba(255, 152, 0, 0.1)',   # Light orange
        'conservative_conservative': 'rgba(244, 67, 54, 0.1)' # Light red
    }
    
    # Draw quadrant backgrounds
    for i, (quad, color) in enumerate(quadrant_colors.items()):
        if quad == 'liberal_liberal':
            x0, x1, y0, y1 = -1, 0, -1, 0
        elif quad == 'liberal_conservative':
            x0, x1, y0, y1 = -1, 0, 0, 1
        elif quad == 'conservative_liberal':
            x0, x1, y0, y1 = 0, 1, -1, 0
        else:  # conservative_conservative
            x0, x1, y0, y1 = 0, 1, 0, 1
        
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            fillcolor=color,
            line=dict(width=0),
        )
    
    # Add axis lines
    fig.add_hline(y=0, line_width=2, line_color="black", opacity=0.5)
    fig.add_vline(x=0, line_width=2, line_color="black", opacity=0.5)
    
    # Add center point
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=8, color='black', symbol='cross'),
        name='Political Center',
        showlegend=True
    ))
    
    # Plot analysis points
    if st.session_state.analysis_history:
        valid_analyses = [a for a in st.session_state.analysis_history if 'error' not in a]
        
        if valid_analyses:
            x_coords = [a['economic_score'] for a in valid_analyses]
            y_coords = [a['social_score'] for a in valid_analyses]
            texts = [f"Analysis {i+1}: {a['text'][:50]}..." for i, a in enumerate(valid_analyses)]
            colors = [a['ideological_intensity'] for a in valid_analyses]
            
            # Plot all points except the latest
            if len(valid_analyses) > 1:
                fig.add_trace(go.Scatter(
                    x=x_coords[:-1], y=y_coords[:-1],
                    mode='markers+text',
                    marker=dict(
                        size=10,
                        color=colors[:-1],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Ideological<br>Intensity"),
                        line=dict(width=1, color='black')
                    ),
                    text=[str(i+1) for i in range(len(valid_analyses)-1)],
                    textposition="middle center",
                    textfont=dict(color="white", size=8),
                    name='Previous Analyses',
                    hovertext=texts[:-1],
                    hoverinfo='text'
                ))
            
            # Highlight the latest point
            latest = valid_analyses[-1]
            fig.add_trace(go.Scatter(
                x=[latest['economic_score']], y=[latest['social_score']],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                text=['LATEST'],
                textposition="top center",
                textfont=dict(color="darkred", size=10, family="Arial Black"),
                name='Latest Analysis',
                hovertext=f"Latest: {latest['text'][:100]}...",
                hoverinfo='text'
            ))
    
    # Customize layout
    fig.update_layout(
        title={
            'text': "Political Compass Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis=dict(
            title="Economic Axis: Liberal ← → Conservative",
            range=[-1.1, 1.1],
            tickmode='linear',
            tick0=-1,
            dtick=0.2,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)'
        ),
        yaxis=dict(
            title="Social Axis: Liberal ← → Conservative",
            range=[-1.1, 1.1],
            tickmode='linear',
            tick0=-1,
            dtick=0.2,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)'
        ),
        width=700,
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        plot_bgcolor='white'
    )
    
    # Add quadrant labels as annotations
    annotations = [
        dict(x=-0.5, y=0.5, text="Liberal Economic<br>Conservative Social", 
             showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        dict(x=0.5, y=0.5, text="Conservative Economic<br>Conservative Social", 
             showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        dict(x=-0.5, y=-0.5, text="Liberal Economic<br>Liberal Social", 
             showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        dict(x=0.5, y=-0.5, text="Conservative Economic<br>Liberal Social", 
             showarrow=False, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)")
    ]
    
    fig.update_layout(annotations=annotations)
    
    return fig

def display_analysis_results(result):
    """Display analysis results in a formatted way"""
    if 'error' in result:
        st.error(f"Analysis Error: {result['error']}")
        return
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Economic Score",
            value=f"{result['economic_score']:.3f}",
            help="Range: -1 (Liberal) to +1 (Conservative)"
        )
        st.caption(result['economic_interpretation'])
    
    with col2:
        st.metric(
            label="Social Score", 
            value=f"{result['social_score']:.3f}",
            help="Range: -1 (Liberal) to +1 (Conservative)"
        )
        st.caption(result['social_interpretation'])
    
    with col3:
        st.metric(
            label="Ideological Intensity",
            value=f"{result['ideological_intensity']:.1f}%",
            help="Distance from political center (0-100%)"
        )
        st.caption(f"Quadrant: {result['quadrant']}")

def main():
    """Main Streamlit app"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🧭 Political Compass Analyzer")
    st.markdown("Analyze political text using your fine-tuned DistilBERT model and visualize results on a 2D political compass")
    
    # Check if model is available
    if not MODEL_AVAILABLE or not st.session_state.model_loaded:
        st.error("Fine-tuned BERT model not available. Please check your installation and model files.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Analysis Options")
        
        # Model info
        st.info(f"Model: DeepSeekPoliticalScorer")
        st.info(f"Analyses performed: {len(st.session_state.analysis_history)}")
        
        # Clear history button
        if st.button("Clear Analysis History", type="secondary"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
            st.rerun()
        
        # Export results
        if st.session_state.analysis_history:
            if st.button("Export Results to CSV"):
                # Create DataFrame
                export_data = []
                for i, analysis in enumerate(st.session_state.analysis_history):
                    if 'error' not in analysis:
                        export_data.append({
                            'analysis_id': i + 1,
                            'text': analysis['text'][:100] + '...' if len(analysis['text']) > 100 else analysis['text'],
                            'economic_score': analysis['economic_score'],
                            'social_score': analysis['social_score'],
                            'economic_interpretation': analysis['economic_interpretation'],
                            'social_interpretation': analysis['social_interpretation'],
                            'quadrant': analysis['quadrant'],
                            'ideological_intensity': analysis['ideological_intensity'],
                            'timestamp': analysis['timestamp']
                        })
                
                if export_data:
                    df = pd.DataFrame(export_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"political_compass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Example texts from your test script
        st.header("Example Texts")
        examples = {
            "Liberal Economic + Liberal Social": "We must expand Medicare for All, guarantee universal basic income, strengthen labor unions, and ensure equal rights for all LGBTQ+ individuals regardless of immigration status.",
            "Conservative Economic + Conservative Social": "We believe in fiscal responsibility, traditional family values, the sanctity of life, limited government spending, and the importance of law and order in our communities.",
            "Project 2025 Immigration": "We must carry out the largest domestic deportation operation in American history, using military forces for border security, ending birthright citizenship, and dismantling the asylum system to protect American sovereignty.",
            "Project 2025 DEI Elimination": "Federal agencies must eliminate all diversity, equity, and inclusion programs that constitute illegal discrimination, restore merit-based opportunity, and prosecute anti-white racism instead of promoting racial preferences.",
            "Project 2025 Abortion": "The FDA must reverse approval of abortion medication mifepristone, enforce the Comstock Act against mail-order abortion pills, and create a national abortion database to track procedures across all states.",
            "Liberal Economic + Conservative Social": "Government should provide universal healthcare and robust social services, while maintaining strong border security and supporting law enforcement in our communities.",
            "Conservative Economic + Liberal Social": "We believe in free market solutions and limited government intervention in the economy, while supporting marriage equality, criminal justice reform, and comprehensive immigration reform.",
            "Moderate Position": "We support a balanced approach to fiscal policy, reasonable regulations on business, bipartisan solutions to healthcare, and pragmatic immigration policies."
        }
        
        selected_example = st.selectbox("Choose an example:", [""] + list(examples.keys()))
        
        if selected_example and st.button("Use Example"):
            st.session_state.example_text = examples[selected_example]
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Text Analysis")
        
        # Text input
        default_text = getattr(st.session_state, 'example_text', '')
        user_text = st.text_area(
            "Enter political text to analyze:",
            value=default_text,
            height=150,
            placeholder="Enter a political statement, policy position, or quote..."
        )
        
        # Clear the example text after using it
        if hasattr(st.session_state, 'example_text'):
            del st.session_state.example_text
        
        # Analyze button
        if st.button("Analyze Text", type="primary", disabled=not user_text.strip()):
            if user_text.strip():
                with st.spinner("Analyzing political position..."):
                    result = analyze_text(user_text)
                    st.session_state.analysis_history.append(result)
                
                # Display results
                st.subheader("Analysis Results")
                display_analysis_results(result)
                
                # Show the text being analyzed
                with st.expander("Analyzed Text"):
                    st.write(user_text)
    
    with col2:
        st.header("Political Compass")
        
        # Create and display compass
        if st.session_state.analysis_history:
            compass_fig = create_plotly_compass()
            st.plotly_chart(compass_fig, use_container_width=True)
        else:
            st.info("No analyses yet. Analyze some text to see it plotted on the compass!")
    
    # Analysis history
    if st.session_state.analysis_history:
        st.header("Analysis History")
        
        # Create DataFrame for display
        history_data = []
        for i, analysis in enumerate(st.session_state.analysis_history):
            if 'error' not in analysis:
                history_data.append({
                    'ID': i + 1,
                    'Text Preview': analysis['text'][:100] + '...' if len(analysis['text']) > 100 else analysis['text'],
                    'Economic': f"{analysis['economic_score']:.3f}",
                    'Social': f"{analysis['social_score']:.3f}",
                    'Economic Interpretation': analysis['economic_interpretation'],
                    'Social Interpretation': analysis['social_interpretation'],
                    'Quadrant': analysis['quadrant'],
                    'Intensity': f"{analysis['ideological_intensity']:.1f}%",
                    'Time': analysis['timestamp'].strftime('%H:%M:%S')
                })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            valid_analyses = [a for a in st.session_state.analysis_history if 'error' not in a]
            if valid_analyses:
                st.subheader("Summary Statistics")
                
                economic_scores = [a['economic_score'] for a in valid_analyses]
                social_scores = [a['social_score'] for a in valid_analyses]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyses", len(valid_analyses))
                with col2:
                    st.metric("Avg Economic Score", f"{np.mean(economic_scores):.3f}")
                with col3:
                    st.metric("Avg Social Score", f"{np.mean(social_scores):.3f}")
                with col4:
                    st.metric("Avg Intensity", f"{np.mean([a['ideological_intensity'] for a in valid_analyses]):.1f}%")
                
                # Quadrant distribution
                quadrants = [a['quadrant'] for a in valid_analyses]
                quadrant_counts = pd.Series(quadrants).value_counts()
                
                if not quadrant_counts.empty:
                    st.subheader("Quadrant Distribution")
                    st.bar_chart(quadrant_counts)
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Try analyzing different types of political statements to see how they map onto the political compass!")
    st.markdown("🧪 **Note**: This uses your fine-tuned DistilBERT model from the test script")

if __name__ == "__main__":
    main()
