"""
📊 Analytics Dashboard - Political Tweet Visualizations

Creates comprehensive visualizations and analytics for political tweet intensity data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from collections import Counter
import numpy as np

class AnalyticsDashboard:
    """Analytics dashboard for political tweet intensity analysis"""
    
    def __init__(self, tweet_tracker):
        self.tweet_tracker = tweet_tracker
        
        # Color schemes for political visualization
        self.colors = {
            'democratic': '#1f4788',
            'republican': '#c41e3a',
            'neutral': '#6c757d',
            'intensity_low': '#28a745',
            'intensity_medium': '#ffc107',
            'intensity_high': '#fd7e14',
            'intensity_extreme': '#dc3545'
        }
        
        self.intensity_colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    
    def show_dashboard(self):
        """Display the complete analytics dashboard"""
        st.title("📊 Political Tweet Analytics Dashboard")
        st.markdown("*Comprehensive analysis of political sentiment and intensity trends*")
        
        # Get analytics data
        analytics_data = self.tweet_tracker.get_analytics_data(days=30)
        
        if not analytics_data:
            self._show_empty_dashboard()
            return
        
        # Dashboard controls
        self._show_dashboard_controls()
        
        # Key metrics overview
        self._show_key_metrics(analytics_data)
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._show_intensity_distribution(analytics_data)
            self._show_political_breakdown(analytics_data)
        
        with col2:
            self._show_timeline_trends(analytics_data)
            self._show_baseline_comparison(analytics_data)
        
        # Detailed analysis sections
        self._show_extreme_tweets_analysis(analytics_data)
        self._show_temporal_patterns(analytics_data)
        self._show_comparative_analysis(analytics_data)
    
    def _show_empty_dashboard(self):
        """Show empty state when no data is available"""
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 16px; margin: 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #495057; margin-bottom: 1rem;">Political Analytics Dashboard</h3>
            <p style="color: #6c757d; font-size: 1.1rem; margin-bottom: 1.5rem;">No tweet data available yet!</p>
            <p style="color: #6c757d;">Analyze some tweets in the main section to see comprehensive analytics here.</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem;">
                <div style="padding: 1rem; background: rgba(31, 71, 136, 0.1); border-radius: 8px;">📈 Intensity trends</div>
                <div style="padding: 1rem; background: rgba(196, 30, 58, 0.1); border-radius: 8px;">🗳️ Political breakdown</div>
                <div style="padding: 1rem; background: rgba(108, 117, 125, 0.1); border-radius: 8px;">⏰ Temporal patterns</div>
                <div style="padding: 1rem; background: rgba(40, 167, 69, 0.1); border-radius: 8px;">🎯 Extreme content analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _show_dashboard_controls(self):
        """Show dashboard control options"""
        with st.expander("🔧 Dashboard Controls", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                time_period = st.selectbox(
                    "Time Period",
                    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
                    index=1
                )
            
            with col2:
                chart_type = st.selectbox(
                    "Chart Style",
                    ["Standard", "Dark Theme", "Colorblind Friendly"],
                    index=0
                )
            
            with col3:
                if st.button("📥 Export Data"):
                    self._export_dashboard_data()
            
            with col4:
                if st.button("🔄 Refresh Analytics"):
                    self.tweet_tracker._clear_analytics_cache()
                    st.rerun()
            
            # Update analytics based on selection
            if time_period != "All Time":
                days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
                # You could update the analytics data here based on selection
    
    def _show_key_metrics(self, analytics_data: Dict[str, Any]):
        """Show key metrics overview"""
        st.markdown("### 📈 Key Metrics Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_tweets = analytics_data.get('total_tweets', 0)
            st.metric("Total Tweets", total_tweets)
        
        with col2:
            avg_intensity = analytics_data.get('avg_intensity', 0)
            st.metric("Avg Intensity", f"{avg_intensity:.1f}/100")
        
        with col3:
            avg_confidence = analytics_data.get('avg_confidence', 0)
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col4:
            political_breakdown = analytics_data.get('political_breakdown', {})
            dem_pct = (political_breakdown.get('Democratic', 0) / total_tweets * 100) if total_tweets > 0 else 0
            st.metric("Democratic %", f"{dem_pct:.1f}%")
        
        with col5:
            intensity_dist = analytics_data.get('intensity_distribution', {})
            extreme_pct = (intensity_dist.get('extreme', 0) / total_tweets * 100) if total_tweets > 0 else 0
            st.metric("Extreme Content", f"{extreme_pct:.1f}%")
    
    def _show_intensity_distribution(self, analytics_data: Dict[str, Any]):
        """Show intensity level distribution"""
        st.markdown("#### 🔥 Intensity Distribution")
        
        intensity_dist = analytics_data.get('intensity_distribution', {})
        
        if not intensity_dist:
            st.info("No intensity data available")
            return
        
        # Create donut chart
        labels = ['Low (0-30)', 'Moderate (30-60)', 'High (60-80)', 'Extreme (80-100)']
        values = [
            intensity_dist.get('low', 0),
            intensity_dist.get('moderate', 0),
            intensity_dist.get('high', 0),
            intensity_dist.get('extreme', 0)
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=self.intensity_colors
        )])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=10
        )
        
        fig.update_layout(
            title="Political Intensity Levels",
            title_x=0.5,
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        total_tweets = sum(values)
        if total_tweets > 0:
            extreme_pct = (values[3] / total_tweets) * 100
            if extreme_pct > 20:
                st.warning(f"⚠️ High proportion of extreme content ({extreme_pct:.1f}%)")
            elif extreme_pct < 5:
                st.success(f"✅ Low extreme content ({extreme_pct:.1f}%)")
            else:
                st.info(f"📊 Moderate extreme content ({extreme_pct:.1f}%)")
    
    def _show_political_breakdown(self, analytics_data: Dict[str, Any]):
        """Show political lean breakdown with intensity overlay"""
        st.markdown("#### 🗳️ Political Breakdown")
        
        political_breakdown = analytics_data.get('political_breakdown', {})
        avg_intensity_by_lean = analytics_data.get('avg_intensity_by_lean', {})
        
        if not political_breakdown:
            st.info("No political breakdown data available")
            return
        
        # Create grouped bar chart
        leans = list(political_breakdown.keys())
        counts = list(political_breakdown.values())
        intensities = [avg_intensity_by_lean.get(lean, 0) for lean in leans]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for counts
        colors = [self.colors['democratic'] if lean == 'Democratic' else self.colors['republican'] 
                 for lean in leans]
        
        fig.add_trace(
            go.Bar(x=leans, y=counts, name="Tweet Count", marker_color=colors),
            secondary_y=False,
        )
        
        # Line chart for intensity
        fig.add_trace(
            go.Scatter(x=leans, y=intensities, mode='lines+markers', name="Avg Intensity", 
                      line=dict(color='orange', width=3), marker=dict(size=8)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Political Lean")
        fig.update_yaxes(title_text="Number of Tweets", secondary_y=False)
        fig.update_yaxes(title_text="Average Intensity", secondary_y=True)
        
        fig.update_layout(
            title="Tweet Count and Average Intensity by Political Lean",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_timeline_trends(self, analytics_data: Dict[str, Any]):
        """Show timeline trends of political intensity"""
        st.markdown("#### 📈 Timeline Trends")
        
        daily_counts = analytics_data.get('daily_counts', {})
        
        if not daily_counts:
            st.info("No timeline data available")
            return
        
        # Convert to proper format for plotting
        try:
            dates = [pd.to_datetime(date) for date in daily_counts.keys()]
            counts = list(daily_counts.values())
            
            # Get trend data from tracker for intensity
            trend_data = self.tweet_tracker.get_trend_analysis()
            
            if trend_data:
                df = pd.DataFrame(trend_data)
                
                # Create subplot with tweet volume and intensity
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Tweet volume
                fig.add_trace(
                    go.Bar(x=df['date'], y=df['tweet_count'], name="Tweet Volume", 
                          marker_color='lightblue', opacity=0.7),
                    secondary_y=False,
                )
                
                # Average intensity line
                fig.add_trace(
                    go.Scatter(x=df['date'], y=df['avg_intensity'], mode='lines+markers', 
                              name="Avg Intensity", line=dict(color='red', width=2)),
                    secondary_y=True,
                )
                
                # Add baseline reference
                fig.add_hline(y=65, line_dash="dash", line_color="gray", 
                             annotation_text="2021 Baseline", secondary_y=True)
                
                fig.update_xaxes(title_text="Date")
                fig.update_yaxes(title_text="Tweet Volume", secondary_y=False)
                fig.update_yaxes(title_text="Intensity Level", secondary_y=True)
                
                fig.update_layout(
                    title="Tweet Volume and Intensity Over Time",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple volume chart if no trend data
                fig = go.Figure(data=[go.Bar(x=dates, y=counts, marker_color='lightblue')])
                fig.update_layout(
                    title="Tweet Volume Over Time",
                    xaxis_title="Date",
                    yaxis_title="Number of Tweets",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error creating timeline chart: {e}")
    
    def _show_baseline_comparison(self, analytics_data: Dict[str, Any]):
        """Show comparison to 2021 baseline"""
        st.markdown("#### ⚖️ Baseline Comparison")
        
        baseline_comparison = analytics_data.get('baseline_comparison', {})
        
        if not baseline_comparison:
            st.info("No baseline comparison data available")
            return
        
        # Create horizontal bar chart
        categories = list(baseline_comparison.keys())
        values = list(baseline_comparison.values())
        
        # Color code based on extremeness
        colors = []
        for cat in categories:
            if 'much more extreme' in cat.lower():
                colors.append(self.colors['intensity_extreme'])
            elif 'more extreme' in cat.lower():
                colors.append(self.colors['intensity_high'])
            elif 'similar' in cat.lower():
                colors.append(self.colors['intensity_medium'])
            else:
                colors.append(self.colors['intensity_low'])
        
        fig = go.Figure(data=[go.Bar(
            y=categories,
            x=values,
            orientation='h',
            marker_color=colors,
            text=values,
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Comparison to 2021 Senator Baseline",
            xaxis_title="Number of Tweets",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        total_tweets = sum(values)
        extreme_tweets = sum(v for k, v in baseline_comparison.items() if 'more extreme' in k.lower())
        
        if total_tweets > 0:
            extreme_pct = (extreme_tweets / total_tweets) * 100
            if extreme_pct > 50:
                st.warning(f"🚨 {extreme_pct:.1f}% of tweets are more extreme than 2021 senators")
            elif extreme_pct > 25:
                st.info(f"📊 {extreme_pct:.1f}% of tweets are more extreme than 2021 senators")
            else:
                st.success(f"✅ Only {extreme_pct:.1f}% of tweets are more extreme than 2021 senators")
    
    def _show_extreme_tweets_analysis(self, analytics_data: Dict[str, Any]):
        """Show analysis of extreme tweets"""
        st.markdown("### 🎯 Extreme Content Analysis")
        
        most_extreme = analytics_data.get('most_extreme_tweets', [])
        most_neutral = analytics_data.get('most_neutral_tweets', [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔥 Most Intense Tweets")
            if most_extreme:
                for i, tweet in enumerate(most_extreme[:3], 1):
                    intensity = tweet.get('intensity_scale', 0)
                    lean = tweet.get('political_lean', 'Unknown')
                    text = tweet.get('tweet_text', '')[:100] + "..." if len(tweet.get('tweet_text', '')) > 100 else tweet.get('tweet_text', '')
                    
                    lean_color = self.colors['democratic'] if lean == 'Democratic' else self.colors['republican']
                    
                    st.markdown(f"""
                    <div style="
                        padding: 1rem; 
                        margin: 0.5rem 0; 
                        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1), rgba(253, 126, 20, 0.05));
                        border-radius: 8px; 
                        border-left: 4px solid #dc3545;
                    ">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <strong>#{i} - Intensity: {intensity:.1f}</strong>
                            <span style="color: {lean_color}; font-weight: 600;">{lean}</span>
                        </div>
                        <div style="font-style: italic; color: #495057;">"{text}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No extreme tweets data available")
        
        with col2:
            st.markdown("#### ✅ Most Neutral Tweets")
            if most_neutral:
                for i, tweet in enumerate(most_neutral[:3], 1):
                    intensity = tweet.get('intensity_scale', 0)
                    lean = tweet.get('political_lean', 'Unknown')
                    text = tweet.get('tweet_text', '')[:100] + "..." if len(tweet.get('tweet_text', '')) > 100 else tweet.get('tweet_text', '')
                    
                    lean_color = self.colors['democratic'] if lean == 'Democratic' else self.colors['republican']
                    
                    st.markdown(f"""
                    <div style="
                        padding: 1rem; 
                        margin: 0.5rem 0; 
                        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1), rgba(255, 193, 7, 0.05));
                        border-radius: 8px; 
                        border-left: 4px solid #28a745;
                    ">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <strong>#{i} - Intensity: {intensity:.1f}</strong>
                            <span style="color: {lean_color}; font-weight: 600;">{lean}</span>
                        </div>
                        <div style="font-style: italic; color: #495057;">"{text}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No neutral tweets data available")
    
    def _show_temporal_patterns(self, analytics_data: Dict[str, Any]):
        """Show temporal patterns analysis"""
        st.markdown("### ⏰ Temporal Patterns")
        
        hourly_pattern = analytics_data.get('hourly_pattern', {})
        day_of_week_pattern = analytics_data.get('day_of_week_pattern', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if hourly_pattern:
                st.markdown("#### 🕐 Hourly Activity")
                
                hours = list(range(24))
                counts = [hourly_pattern.get(hour, 0) for hour in hours]
                
                fig = go.Figure(data=[go.Bar(
                    x=hours,
                    y=counts,
                    marker_color='steelblue',
                    text=counts,
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title="Tweet Activity by Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Tweets",
                    height=300,
                    xaxis=dict(tickmode='linear', tick0=0, dtick=2)
                )
                
                # Add time period annotations
                fig.add_vrect(x0=6, x1=12, fillcolor="rgba(255, 255, 0, 0.1)", line_width=0)
                fig.add_vrect(x0=12, x1=18, fillcolor="rgba(255, 165, 0, 0.1)", line_width=0)
                fig.add_vrect(x0=18, x1=22, fillcolor="rgba(0, 0, 255, 0.1)", line_width=0)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak analysis
                if counts:
                    peak_hour = hours[counts.index(max(counts))]
                    if 6 <= peak_hour <= 9:
                        st.info("🌅 Peak activity during morning hours")
                    elif 12 <= peak_hour <= 14:
                        st.info("☀️ Peak activity during lunch hours")
                    elif 18 <= peak_hour <= 21:
                        st.info("🌆 Peak activity during evening hours")
                    else:
                        st.info(f"🕐 Peak activity at {peak_hour}:00")
        
        with col2:
            if day_of_week_pattern:
                st.markdown("#### 📅 Day of Week Activity")
                
                days = list(day_of_week_pattern.keys())
                counts = list(day_of_week_pattern.values())
                
                fig = go.Figure(data=[go.Bar(
                    x=days,
                    y=counts,
                    marker_color='lightcoral',
                    text=counts,
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title="Tweet Activity by Day of Week",
                    xaxis_title="Day of Week",
                    yaxis_title="Number of Tweets",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekend vs weekday analysis
                if counts and days:
                    weekday_total = sum(c for d, c in zip(days, counts) if d not in ['Saturday', 'Sunday'])
                    weekend_total = sum(c for d, c in zip(days, counts) if d in ['Saturday', 'Sunday'])
                    
                    if weekend_total > weekday_total:
                        st.info("📱 More active on weekends")
                    else:
                        st.info("💼 More active on weekdays")
    
    def _show_comparative_analysis(self, analytics_data: Dict[str, Any]):
        """Show comparative analysis across different metrics"""
        st.markdown("### 📊 Comparative Analysis")
        
        # Word count vs intensity analysis
        word_count_by_intensity = analytics_data.get('word_count_by_intensity', {})
        avg_word_count = analytics_data.get('avg_word_count', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Word Count vs Intensity")
            
            if word_count_by_intensity:
                # Convert interval keys to readable labels
                intensity_labels = []
                word_counts = []
                
                for interval, count in word_count_by_intensity.items():
                    if not pd.isna(count):  # Skip NaN values
                        # Convert interval to readable string
                        intensity_labels.append(str(interval))
                        word_counts.append(count)
                
                if intensity_labels and word_counts:
                    fig = go.Figure(data=[go.Bar(
                        x=intensity_labels,
                        y=word_counts,
                        marker_color='purple',
                        text=[f"{wc:.1f}" for wc in word_counts],
                        textposition='outside'
                    )])
                    
                    fig.update_layout(
                        title="Average Word Count by Intensity Level",
                        xaxis_title="Intensity Range",
                        yaxis_title="Average Word Count",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analysis
                    if len(word_counts) > 1:
                        highest_wc = max(word_counts)
                        lowest_wc = min(word_counts)
                        if highest_wc > lowest_wc * 1.2:
                            st.info("📈 Higher intensity tweets tend to be longer")
                        else:
                            st.info("📊 Word count relatively consistent across intensity levels")
                else:
                    st.info("No word count data available")
            else:
                st.info("No word count analysis data available")
        
        with col2:
            st.markdown("#### 🎯 Summary Statistics")
            
            total_tweets = analytics_data.get('total_tweets', 0)
            avg_intensity = analytics_data.get('avg_intensity', 0)
            std_intensity = analytics_data.get('std_intensity', 0)
            
            st.markdown(f"""
            **📊 Dataset Overview:**
            - **Total Tweets Analyzed:** {total_tweets:,}
            - **Average Intensity:** {avg_intensity:.2f}/100
            - **Intensity Std Dev:** {std_intensity:.2f}
            - **Average Word Count:** {avg_word_count:.1f}
            
            **🎯 Key Insights:**
            """)
            
            # Generate insights based on data
            insights = []
            
            if avg_intensity > 60:
                insights.append("⚠️ High average intensity detected")
            elif avg_intensity < 30:
                insights.append("✅ Low average intensity - neutral discourse")
            else:
                insights.append("📊 Moderate intensity levels")
            
            if std_intensity > 25:
                insights.append("📈 High variability in content intensity")
            else:
                insights.append("📊 Consistent intensity levels")
            
            political_breakdown = analytics_data.get('political_breakdown', {})
            if political_breakdown:
                dem_count = political_breakdown.get('Democratic', 0)
                rep_count = political_breakdown.get('Republican', 0)
                total_political = dem_count + rep_count
                
                if total_political > 0:
                    dem_pct = (dem_count / total_political) * 100
                    if abs(dem_pct - 50) > 20:
                        dominant = "Democratic" if dem_pct > 50 else "Republican"
                        insights.append(f"🗳️ {dominant} content dominates ({dem_pct:.1f}% vs {100-dem_pct:.1f}%)")
                    else:
                        insights.append("⚖️ Balanced Democratic/Republican content")
            
            for insight in insights:
                st.markdown(f"- {insight}")
    
    def _export_dashboard_data(self):
        """Export dashboard data"""
        try:
            filename = self.tweet_tracker.export_data('json')
            if filename:
                st.success(f"✅ Dashboard data exported to: {filename}")
                
                # Provide download link
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="📥 Download Export File",
                        data=f.read(),
                        file_name=filename,
                        mime="application/json"
                    )
            else:
                st.error("❌ Export failed")
                
        except Exception as e:
            st.error(f"❌ Export error: {e}")
    
    def create_intensity_heatmap(self, analytics_data: Dict[str, Any]) -> go.Figure:
        """Create a heatmap of intensity patterns"""
        hourly_pattern = analytics_data.get('hourly_pattern', {})
        day_of_week_pattern = analytics_data.get('day_of_week_pattern', {})
        
        if not hourly_pattern or not day_of_week_pattern:
            return None
        
        # This would require more detailed temporal data
        # For now, return a simple placeholder
        fig = go.Figure(data=go.Heatmap(
            z=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            x=['Low', 'Medium', 'High'],
            y=['Weekday', 'Weekend', 'All Days'],
            colorscale='RdYlBu_r'
        ))
        
        fig.update_layout(
            title="Political Intensity Heatmap",
            height=300
        )
        
        return fig
    
    def generate_insights_report(self, analytics_data: Dict[str, Any]) -> str:
        """Generate a text report of key insights"""
        if not analytics_data:
            return "No data available for analysis."
        
        total_tweets = analytics_data.get('total_tweets', 0)
        avg_intensity = analytics_data.get('avg_intensity', 0)
        political_breakdown = analytics_data.get('political_breakdown', {})
        baseline_comparison = analytics_data.get('baseline_comparison', {})
        
        report = f"""
# Political Tweet Intensity Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Tweets Analyzed:** {total_tweets:,}
- **Average Intensity:** {avg_intensity:.2f}/100
- **Analysis Period:** Last 30 days

## Political Breakdown
"""
        
        for lean, count in political_breakdown.items():
            percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
            report += f"- **{lean}:** {count} tweets ({percentage:.1f}%)\n"
        
        report += "\n## Baseline Comparison\n"
        for comparison, count in baseline_comparison.items():
            percentage = (count / total_tweets * 100) if total_tweets > 0 else 0
            report += f"- **{comparison}:** {count} tweets ({percentage:.1f}%)\n"
        
        # Add key insights
        report += "\n## Key Insights\n"
        
        if avg_intensity > 70:
            report += "- ⚠️ **High intensity discourse detected** - content significantly more extreme than baseline\n"
        elif avg_intensity < 40:
            report += "- ✅ **Moderate discourse intensity** - content within normal political range\n"
        
        extreme_count = sum(count for desc, count in baseline_comparison.items() if 'more extreme' in desc.lower())
        extreme_pct = (extreme_count / total_tweets * 100) if total_tweets > 0 else 0
        
        if extreme_pct > 50:
            report += f"- 🚨 **Majority of content exceeds 2021 baseline** ({extreme_pct:.1f}%)\n"
        elif extreme_pct < 20:
            report += f"- 📊 **Most content similar to 2021 baseline** (only {extreme_pct:.1f}% more extreme)\n"
        
        return report
    
    def create_summary_cards(self, analytics_data: Dict[str, Any]):
        """Create summary cards for quick overview"""
        total_tweets = analytics_data.get('total_tweets', 0)
        avg_intensity = analytics_data.get('avg_intensity', 0)
        intensity_dist = analytics_data.get('intensity_distribution', {})
        
        # Calculate key percentages
        extreme_pct = (intensity_dist.get('extreme', 0) / total_tweets * 100) if total_tweets > 0 else 0
        high_pct = (intensity_dist.get('high', 0) / total_tweets * 100) if total_tweets > 0 else 0
        
        cards_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1rem 0;">
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #1f4788;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1f4788;">📊 Total Analysis</h4>
                <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #333;">{total_tweets:,}</p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Tweets Analyzed</p>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #ffc107;">
                <h4 style="margin: 0 0 0.5rem 0; color: #f57c00;">🔥 Avg Intensity</h4>
                <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #333;">{avg_intensity:.1f}/100</p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Political Intensity</p>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #dc3545;">
                <h4 style="margin: 0 0 0.5rem 0; color: #c41e3a;">⚠️ Extreme Content</h4>
                <p style="font-size: 2rem; font-weight: bold; margin: 0; color: #333;">{extreme_pct:.1f}%</p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">High Intensity Tweets</p>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #28a745;">
                <h4 style="margin: 0 0 0.5rem 0; color: #155724;">📈 Trend Status</h4>
                <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #333;">
                    {'🔴 High' if extreme_pct > 20 else '🟡 Moderate' if extreme_pct > 10 else '🟢 Low'}
                </p>
                <p style="margin: 0.5rem 0 0 0; color: #666;">Extremism Level</p>
            </div>
        </div>
        """
        
        return cards_html