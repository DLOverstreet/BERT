"""
📊 Tweet Tracker - Data Storage and Analytics

Handles storage, retrieval, and analysis of political tweet data with privacy controls.
"""

import os
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import Counter, defaultdict
import hashlib

# Try to import pandas, fallback if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class TweetTracker:
    """Enhanced tweet tracker with analytics and privacy controls"""
    
    def __init__(self, data_dir: str = "tweet_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Data files
        self.tweets_file = self.data_dir / "analyzed_tweets.csv"
        self.analytics_file = self.data_dir / "analytics_cache.json"
        self.settings_file = self.data_dir / "settings.json"
        
        # Initialize files
        self._init_csv_file()
        self._load_settings()
        
        # Analytics cache
        self._analytics_cache = {}
        self._cache_timestamp = None
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not self.tweets_file.exists():
            headers = [
                'timestamp', 'tweet_text', 'political_lean', 'dem_score', 'rep_score',
                'partisan_intensity', 'confidence', 'extremism_score', 'intensity_scale',
                'vs_baseline', 'intensity_percentile', 'session_id', 'anonymized_hash',
                'word_count', 'model_version'
            ]
            
            with open(self.tweets_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def _load_settings(self):
        """Load settings from file or create defaults"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
            except:
                self.settings = self._default_settings()
        else:
            self.settings = self._default_settings()
            self._save_settings()
    
    def _default_settings(self) -> Dict[str, Any]:
        """Default application settings"""
        return {
            'data_retention_days': 365,
            'anonymize_after_days': 30,
            'cache_duration_minutes': 15,
            'max_tweets_display': 100,
            'privacy_mode': False,
            'export_format': 'csv',
            'created_at': datetime.now().isoformat()
        }
    
    def _save_settings(self):
        """Save settings to file"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def _anonymize_session_id(self, session_id: str) -> str:
        """Create anonymized hash of session ID"""
        return hashlib.sha256(session_id.encode()).hexdigest()[:12]
    
    def log_tweet(self, tweet_text: str, analysis_result: Dict[str, Any], session_id: str = "anonymous"):
        """Log a tweet analysis result"""
        try:
            timestamp = datetime.now().isoformat()
            anonymized_hash = self._anonymize_session_id(session_id)
            word_count = len(tweet_text.split())
            
            # Extract analysis results
            row_data = [
                timestamp,
                tweet_text,
                analysis_result.get('political_lean', ''),
                analysis_result.get('dem_score', 0),
                analysis_result.get('rep_score', 0),
                analysis_result.get('partisan_intensity', 0),
                analysis_result.get('confidence', 0),
                analysis_result.get('extremism_score', 0),
                analysis_result.get('intensity_scale', 0),
                analysis_result.get('vs_baseline', ''),
                analysis_result.get('intensity_percentile', 0),
                session_id,
                anonymized_hash,
                word_count,
                analysis_result.get('model_version', 'unknown')
            ]
            
            with open(self.tweets_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            # Clear cache after new data
            self._clear_analytics_cache()
            
            # Cleanup old data periodically
            self._maybe_cleanup_old_data()
            
        except Exception as e:
            print(f"Error logging tweet: {e}")
    
    def _maybe_cleanup_old_data(self):
        """Cleanup old data based on retention settings"""
        # Only cleanup occasionally to avoid performance issues
        import random
        if random.random() < 0.1:  # 10% chance on each log
            self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove old data based on retention settings"""
        if not self.tweets_file.exists() or not PANDAS_AVAILABLE:
            return
        
        try:
            retention_days = self.settings.get('data_retention_days', 365)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            df = pd.read_csv(self.tweets_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Keep only recent data
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            # Anonymize session IDs for older data if needed
            anonymize_after_days = self.settings.get('anonymize_after_days', 30)
            anonymize_cutoff = datetime.now() - timedelta(days=anonymize_after_days)
            
            recent_df.loc[recent_df['timestamp'] < anonymize_cutoff, 'session_id'] = 'anonymized'
            
            # Save cleaned data
            recent_df.to_csv(self.tweets_file, index=False)
            
        except Exception as e:
            print(f"Data cleanup error: {e}")
    
    def get_recent_tweets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweet analyses"""
        if not self.tweets_file.exists():
            return []
        
        try:
            if PANDAS_AVAILABLE:
                df = pd.read_csv(self.tweets_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False).head(limit)
                
                return [
                    {
                        'timestamp': row['timestamp'],
                        'tweet_text': row['tweet_text'],
                        'political_lean': row['political_lean'],
                        'intensity_scale': row['intensity_scale'],
                        'confidence': row['confidence'],
                        'vs_baseline': row['vs_baseline'],
                        'extremism_score': row['extremism_score']
                    }
                    for _, row in df.iterrows()
                ]
            else:
                # Fallback without pandas
                tweets = []
                with open(self.tweets_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            tweets.append({
                                'timestamp': datetime.fromisoformat(row['timestamp']),
                                'tweet_text': row['tweet_text'],
                                'political_lean': row['political_lean'],
                                'intensity_scale': float(row['intensity_scale']),
                                'confidence': float(row['confidence']),
                                'vs_baseline': row['vs_baseline'],
                                'extremism_score': float(row['extremism_score'])
                            })
                        except:
                            continue
                
                # Sort and limit
                tweets.sort(key=lambda x: x['timestamp'], reverse=True)
                return tweets[:limit]
                
        except Exception as e:
            print(f"Error getting recent tweets: {e}")
            return []
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the sidebar"""
        try:
            if self._is_cache_valid():
                return self._analytics_cache.get('summary_stats', {})
            
            if not self.tweets_file.exists():
                return {}
            
            if PANDAS_AVAILABLE:
                df = pd.read_csv(self.tweets_file)
                if len(df) == 0:
                    return {}
                
                # Calculate recent stats (last 7 days)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                recent_cutoff = datetime.now() - timedelta(days=7)
                recent_df = df[df['timestamp'] >= recent_cutoff]
                
                if len(recent_df) == 0:
                    recent_df = df.tail(10)  # Use last 10 if no recent data
                
                stats = {
                    'total_tweets': len(df),
                    'recent_tweets': len(recent_df),
                    'avg_intensity': recent_df['intensity_scale'].mean(),
                    'max_intensity': recent_df['intensity_scale'].max(),
                    'min_intensity': recent_df['intensity_scale'].min(),
                    'avg_confidence': recent_df['confidence'].mean(),
                    'most_extreme_direction': recent_df.loc[recent_df['intensity_scale'].idxmax(), 'political_lean'] if len(recent_df) > 0 else None,
                    'dem_percentage': (recent_df['political_lean'] == 'Democratic').mean() * 100,
                    'rep_percentage': (recent_df['political_lean'] == 'Republican').mean() * 100
                }
                
                self._analytics_cache['summary_stats'] = stats
                self._cache_timestamp = datetime.now()
                
                return stats
            else:
                # Basic stats without pandas
                return self._get_basic_summary_stats()
                
        except Exception as e:
            print(f"Error getting summary stats: {e}")
            return {}
    
    def _get_basic_summary_stats(self) -> Dict[str, Any]:
        """Get basic summary stats without pandas"""
        try:
            tweets = []
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tweets.append({
                            'timestamp': datetime.fromisoformat(row['timestamp']),
                            'intensity_scale': float(row['intensity_scale']),
                            'confidence': float(row['confidence']),
                            'political_lean': row['political_lean']
                        })
                    except:
                        continue
            
            if not tweets:
                return {}
            
            # Recent tweets (last 7 days)
            recent_cutoff = datetime.now() - timedelta(days=7)
            recent_tweets = [t for t in tweets if t['timestamp'] >= recent_cutoff]
            if not recent_tweets:
                recent_tweets = tweets[-10:]  # Last 10 if no recent
            
            intensities = [t['intensity_scale'] for t in recent_tweets]
            confidences = [t['confidence'] for t in recent_tweets]
            leans = [t['political_lean'] for t in recent_tweets]
            
            dem_count = leans.count('Democratic')
            rep_count = leans.count('Republican')
            total_recent = len(recent_tweets)
            
            return {
                'total_tweets': len(tweets),
                'recent_tweets': total_recent,
                'avg_intensity': sum(intensities) / len(intensities) if intensities else 0,
                'max_intensity': max(intensities) if intensities else 0,
                'min_intensity': min(intensities) if intensities else 0,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'most_extreme_direction': leans[intensities.index(max(intensities))] if intensities else None,
                'dem_percentage': (dem_count / total_recent * 100) if total_recent > 0 else 0,
                'rep_percentage': (rep_count / total_recent * 100) if total_recent > 0 else 0
            }
            
        except Exception as e:
            print(f"Error in basic summary stats: {e}")
            return {}
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics data"""
        try:
            cache_key = f'analytics_{days}d'
            if self._is_cache_valid() and cache_key in self._analytics_cache:
                return self._analytics_cache[cache_key]
            
            if not self.tweets_file.exists():
                return {}
            
            if PANDAS_AVAILABLE:
                return self._get_pandas_analytics(days, cache_key)
            else:
                return self._get_basic_analytics(days, cache_key)
                
        except Exception as e:
            print(f"Error getting analytics data: {e}")
            return {}
    
    def _get_pandas_analytics(self, days: int, cache_key: str) -> Dict[str, Any]:
        """Get analytics using pandas"""
        df = pd.read_csv(self.tweets_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to recent data
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_df = df[df['timestamp'] >= cutoff_date]
        
        if len(recent_df) == 0:
            return {}
        
        # Time-based analysis
        recent_df['date'] = recent_df['timestamp'].dt.date
        recent_df['hour'] = recent_df['timestamp'].dt.hour
        recent_df['day_of_week'] = recent_df['timestamp'].dt.day_name()
        
        # Basic metrics
        analytics = {
            'total_tweets': len(recent_df),
            'unique_sessions': recent_df['session_id'].nunique(),
            'avg_intensity': recent_df['intensity_scale'].mean(),
            'std_intensity': recent_df['intensity_scale'].std(),
            'avg_confidence': recent_df['confidence'].mean(),
            
            # Political breakdown
            'political_breakdown': recent_df['political_lean'].value_counts().to_dict(),
            'avg_intensity_by_lean': recent_df.groupby('political_lean')['intensity_scale'].mean().to_dict(),
            
            # Intensity distribution
            'intensity_distribution': {
                'low': len(recent_df[recent_df['intensity_scale'] < 30]),
                'moderate': len(recent_df[(recent_df['intensity_scale'] >= 30) & (recent_df['intensity_scale'] < 60)]),
                'high': len(recent_df[(recent_df['intensity_scale'] >= 60) & (recent_df['intensity_scale'] < 80)]),
                'extreme': len(recent_df[recent_df['intensity_scale'] >= 80])
            },
            
            # Time patterns
            'daily_counts': recent_df.groupby('date').size().to_dict(),
            'hourly_pattern': recent_df.groupby('hour').size().to_dict(),
            'day_of_week_pattern': recent_df.groupby('day_of_week').size().to_dict(),
            
            # Baseline comparisons
            'baseline_comparison': recent_df['vs_baseline'].value_counts().to_dict(),
            
            # Top tweets
            'most_extreme_tweets': recent_df.nlargest(5, 'intensity_scale')[['tweet_text', 'intensity_scale', 'political_lean']].to_dict('records'),
            'most_neutral_tweets': recent_df.nsmallest(5, 'intensity_scale')[['tweet_text', 'intensity_scale', 'political_lean']].to_dict('records'),
            
            # Word analysis
            'avg_word_count': recent_df['word_count'].mean(),
            'word_count_by_intensity': recent_df.groupby(pd.cut(recent_df['intensity_scale'], bins=[0, 30, 60, 80, 100]))['word_count'].mean().to_dict()
        }
        
        # Cache the results
        self._analytics_cache[cache_key] = analytics
        self._cache_timestamp = datetime.now()
        
        return analytics
    
    def _get_basic_analytics(self, days: int, cache_key: str) -> Dict[str, Any]:
        """Get basic analytics without pandas"""
        try:
            tweets = []
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        if timestamp >= cutoff_date:
                            tweets.append({
                                'timestamp': timestamp,
                                'tweet_text': row['tweet_text'],
                                'political_lean': row['political_lean'],
                                'intensity_scale': float(row['intensity_scale']),
                                'confidence': float(row['confidence']),
                                'vs_baseline': row['vs_baseline'],
                                'word_count': int(row.get('word_count', 0))
                            })
                    except:
                        continue
            
            if not tweets:
                return {}
            
            # Calculate basic analytics
            intensities = [t['intensity_scale'] for t in tweets]
            confidences = [t['confidence'] for t in tweets]
            leans = [t['political_lean'] for t in tweets]
            baselines = [t['vs_baseline'] for t in tweets]
            word_counts = [t['word_count'] for t in tweets]
            
            analytics = {
                'total_tweets': len(tweets),
                'avg_intensity': sum(intensities) / len(intensities),
                'avg_confidence': sum(confidences) / len(confidences),
                'political_breakdown': dict(Counter(leans)),
                'baseline_comparison': dict(Counter(baselines)),
                'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
                
                'intensity_distribution': {
                    'low': len([i for i in intensities if i < 30]),
                    'moderate': len([i for i in intensities if 30 <= i < 60]),
                    'high': len([i for i in intensities if 60 <= i < 80]),
                    'extreme': len([i for i in intensities if i >= 80])
                }
            }
            
            # Find most extreme tweets
            sorted_tweets = sorted(tweets, key=lambda x: x['intensity_scale'], reverse=True)
            analytics['most_extreme_tweets'] = [
                {
                    'tweet_text': t['tweet_text'],
                    'intensity_scale': t['intensity_scale'],
                    'political_lean': t['political_lean']
                }
                for t in sorted_tweets[:5]
            ]
            
            # Cache results
            self._analytics_cache[cache_key] = analytics
            self._cache_timestamp = datetime.now()
            
            return analytics
            
        except Exception as e:
            print(f"Error in basic analytics: {e}")
            return {}
    
    def get_trend_analysis(self) -> List[Dict[str, Any]]:
        """Get trend analysis for visualization"""
        try:
            if not self.tweets_file.exists():
                return []
            
            if PANDAS_AVAILABLE:
                df = pd.read_csv(self.tweets_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                
                # Group by date and calculate metrics
                daily_stats = df.groupby('date').agg({
                    'intensity_scale': ['mean', 'max', 'count'],
                    'political_lean': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
                }).reset_index()
                
                daily_stats.columns = ['date', 'avg_intensity', 'max_intensity', 'tweet_count', 'dominant_lean']
                
                return daily_stats.to_dict('records')
            else:
                # Basic trend analysis without pandas
                return self._get_basic_trend_analysis()
                
        except Exception as e:
            print(f"Error getting trend analysis: {e}")
            return []
    
    def _get_basic_trend_analysis(self) -> List[Dict[str, Any]]:
        """Get basic trend analysis without pandas"""
        try:
            daily_data = defaultdict(list)
            
            with open(self.tweets_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        date = timestamp.date()
                        
                        daily_data[date].append({
                            'intensity_scale': float(row['intensity_scale']),
                            'political_lean': row['political_lean']
                        })
                    except:
                        continue
            
            trends = []
            for date, tweets in daily_data.items():
                intensities = [t['intensity_scale'] for t in tweets]
                leans = [t['political_lean'] for t in tweets]
                
                trend = {
                    'date': date,
                    'avg_intensity': sum(intensities) / len(intensities),
                    'max_intensity': max(intensities),
                    'tweet_count': len(tweets),
                    'dominant_lean': max(set(leans), key=leans.count) if leans else 'Unknown'
                }
                trends.append(trend)
            
            return sorted(trends, key=lambda x: x['date'])
            
        except Exception as e:
            print(f"Error in basic trend analysis: {e}")
            return []
    
    def get_period_comparisons(self) -> Dict[str, Dict[str, Any]]:
        """Get comparisons between different time periods"""
        try:
            periods = {
                'Last 7 Days': 7,
                'Last 30 Days': 30,
                'Last 90 Days': 90
            }
            
            comparisons = {}
            for period_name, days in periods.items():
                analytics = self.get_analytics_data(days)
                if analytics:
                    comparisons[period_name] = {
                        'avg_intensity': analytics.get('avg_intensity', 0),
                        'total_tweets': analytics.get('total_tweets', 0),
                        'extreme_count': analytics.get('intensity_distribution', {}).get('extreme', 0)
                    }
            
            return comparisons
            
        except Exception as e:
            print(f"Error getting period comparisons: {e}")
            return {}
    
    def _is_cache_valid(self) -> bool:
        """Check if analytics cache is still valid"""
        if not self._cache_timestamp:
            return False
        
        cache_duration = self.settings.get('cache_duration_minutes', 15)
        return (datetime.now() - self._cache_timestamp).total_seconds() < (cache_duration * 60)
    
    def _clear_analytics_cache(self):
        """Clear the analytics cache"""
        self._analytics_cache = {}
        self._cache_timestamp = None
    
    def export_data(self, format: str = 'csv') -> Optional[str]:
        """Export data to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'csv':
                filename = f"political_tweets_export_{timestamp}.csv"
                if self.tweets_file.exists():
                    import shutil
                    shutil.copy2(self.tweets_file, filename)
                    return filename
            
            elif format.lower() == 'json':
                filename = f"political_tweets_export_{timestamp}.json"
                analytics = self.get_analytics_data(days=365)  # Full year
                
                export_data = {
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'format': 'json',
                        'tool': 'Political Tweet Intensity Analyzer'
                    },
                    'analytics': analytics,
                    'recent_tweets': self.get_recent_tweets(limit=50)
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return filename
            
            return None
            
        except Exception as e:
            print(f"Export failed: {e}")
            return None
    
    def clear_data(self, confirm: bool = False):
        """Clear all stored data"""
        if not confirm:
            return False
        
        try:
            if self.tweets_file.exists():
                self.tweets_file.unlink()
                self._init_csv_file()
            
            self._clear_analytics_cache()
            
            return True
            
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
    
    def update_settings(self, new_settings: Dict[str, Any]):
        """Update application settings"""
        self.settings.update(new_settings)
        self._save_settings()
        self._clear_analytics_cache()  # Clear cache when settings change