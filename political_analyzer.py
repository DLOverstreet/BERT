"""
🤖 Political Tweet Analyzer using DistilBERT

This module handles the core ML analysis using the fine-tuned DistilBERT model
for political sentiment classification and intensity scoring.
"""

import streamlit as st
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoliticalAnalyzer:
    """Political tweet analyzer using DistilBERT model"""
    
    def __init__(self):
        self.model_name = "m-newhauser/distilbert-political-tweets"
        self.model = None
        self.tokenizer = None
        self.model_available = False
        
        # 2021 baseline statistics (estimated from model training data)
        self.baseline_stats = {
            'mean_intensity': 0.65,
            'std_intensity': 0.20,
            'mean_confidence': 0.78
        }
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the DistilBERT model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            logger.info("Starting model download/load...")
            
            # Initialize with pipeline for easier use
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,  # Updated from deprecated return_all_scores=True
                device=-1,   # Use CPU (-1) or GPU (0)
                model_kwargs={"cache_dir": None}  # Use default cache
            )
            
            # Also load tokenizer and model directly for more control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.model_available = True
            logger.info("✅ DistilBERT model loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ Missing dependencies: {e}")
            self.model_available = False
            # Don't show streamlit error here as it might not be available
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model_available = False
            
            # Check if this is a common cloud deployment issue
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                logger.error("This appears to be a network/timeout issue. The model download may have failed.")
            elif "memory" in str(e).lower() or "out of memory" in str(e).lower():
                logger.error("This appears to be a memory issue. The model may be too large for the current environment.")
            else:
                logger.error(f"Unexpected error during model loading: {e}")
    
    def analyze_tweet(self, tweet_text: str) -> Dict[str, Any]:
        """
        Analyze a tweet for political sentiment and intensity
        
        Args:
            tweet_text (str): The tweet text to analyze
            
        Returns:
            Dict containing analysis results
        """
        if not self.model_available:
            return {"error": "Model not available"}
        
        if not tweet_text or len(tweet_text.strip()) < 5:
            return {"error": "Tweet text too short"}
        
        try:
            # Get predictions from the model
            results = self.classifier(tweet_text)
            
            # Debug: Print results to help troubleshoot
            logger.info(f"Model results: {results}")
            
            # Extract scores - handle both old and new format
            dem_score = 0
            rep_score = 0
            
            # Results is a list of dictionaries
            if isinstance(results, list) and len(results) > 0:
                result_list = results  # top_k=None returns list directly
            else:
                logger.error(f"Unexpected results format: {results}")
                return {"error": f"Unexpected model output format: {type(results)}"}
            
            for result in result_list:
                label = result['label'].upper()
                score = result['score']
                
                if 'DEMOCRATIC' in label or 'DEMOCRAT' in label:
                    dem_score = score
                elif 'REPUBLICAN' in label:
                    rep_score = score
            
            # Validate we got both scores
            if dem_score == 0 and rep_score == 0:
                logger.error(f"Failed to extract scores from results: {results}")
                return {"error": "Failed to extract political scores from model output"}
            
            logger.info(f"Extracted scores - Dem: {dem_score}, Rep: {rep_score}")
            
            # Calculate derived metrics
            political_direction = dem_score - rep_score  # -1 to +1
            partisan_intensity = abs(dem_score - rep_score)  # 0 to 1
            confidence = max(dem_score, rep_score) * 100  # 0 to 100
            extremism_score = partisan_intensity * (confidence / 100)  # Combined metric
            
            # Scale intensity to 0-100
            intensity_scale = extremism_score * 100
            
            # Determine political lean
            political_lean = "Democratic" if dem_score > rep_score else "Republican"
            
            # Compare to baseline
            vs_baseline = self._compare_to_baseline(extremism_score)
            
            # Get intensity percentile
            intensity_percentile = self._get_intensity_percentile(extremism_score)
            
            result = {
                'tweet_text': tweet_text,
                'dem_score': dem_score,
                'rep_score': rep_score,
                'political_direction': political_direction,
                'partisan_intensity': partisan_intensity,
                'confidence': confidence,
                'extremism_score': extremism_score,
                'intensity_scale': intensity_scale,
                'political_lean': political_lean,
                'vs_baseline': vs_baseline,
                'intensity_percentile': intensity_percentile,
                'timestamp': datetime.now(),
                'model_version': self.model_name
            }
            
            logger.info(f"Analysis completed successfully: {political_lean}, intensity: {intensity_scale:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _compare_to_baseline(self, score: float) -> str:
        """Compare score to 2021 senator baseline"""
        z_score = (score - self.baseline_stats['mean_intensity']) / self.baseline_stats['std_intensity']
        
        if z_score > 2:
            return "Much more extreme than 2021 senators"
        elif z_score > 1:
            return "More extreme than 2021 senators"
        elif z_score > -1:
            return "Similar to 2021 senators"
        else:
            return "Less extreme than 2021 senators"
    
    def _get_intensity_percentile(self, score: float) -> float:
        """Convert raw score to percentile (0-100)"""
        # Normalize based on observed range and baseline
        # This is a rough calibration - in production you'd want actual percentile data
        percentile = min(100, max(0, score * 120))
        return round(percentile, 1)
    
    def analyze_batch(self, tweets: list) -> list:
        """Analyze multiple tweets at once"""
        if not self.model_available:
            return [{"error": "Model not available"} for _ in tweets]
        
        results = []
        for tweet in tweets:
            if isinstance(tweet, str):
                result = self.analyze_tweet(tweet)
                results.append(result)
            else:
                results.append({"error": "Invalid tweet format"})
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.model_name,
            'model_available': self.model_available,
            'baseline_stats': self.baseline_stats,
            'description': "DistilBERT fine-tuned on 2021 US senator tweets for political sentiment classification"
        }
    
    def validate_input(self, text: str) -> Dict[str, Any]:
        """Validate input text for analysis"""
        if not text:
            return {"valid": False, "error": "No text provided"}
        
        text = text.strip()
        
        if len(text) < 5:
            return {"valid": False, "error": "Text too short (minimum 5 characters)"}
        
        if len(text) > 1000:
            return {"valid": False, "error": "Text too long (maximum 1000 characters)", "truncated": text[:1000]}
        
        # Check for obviously non-political content
        political_keywords = [
            'policy', 'government', 'democrat', 'republican', 'liberal', 'conservative',
            'president', 'congress', 'senate', 'election', 'vote', 'political', 'america',
            'tax', 'healthcare', 'economy', 'immigration', 'climate', 'abortion', 'gun',
            'freedom', 'liberty', 'socialist', 'fascist', 'radical', 'establishment'
        ]
        
        text_lower = text.lower()
        has_political_content = any(keyword in text_lower for keyword in political_keywords)
        
        return {
            "valid": True,
            "text": text,
            "length": len(text),
            "word_count": len(text.split()),
            "likely_political": has_political_content,
            "warning": None if has_political_content else "Text may not contain political content"
        }

class IntensityTracker:
    """Track and compare political intensity over time"""
    
    def __init__(self):
        self.analyses = []
    
    def add_analysis(self, analysis: Dict[str, Any]):
        """Add a new analysis to tracking"""
        if 'error' not in analysis:
            self.analyses.append(analysis)
    
    def get_intensity_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get intensity trends over specified days"""
        if not self.analyses:
            return {}
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_analyses = [
            a for a in self.analyses 
            if a.get('timestamp', datetime.min) >= cutoff_date
        ]
        
        if not recent_analyses:
            return {}
        
        intensities = [a['intensity_scale'] for a in recent_analyses]
        
        return {
            'count': len(recent_analyses),
            'avg_intensity': np.mean(intensities),
            'max_intensity': np.max(intensities),
            'min_intensity': np.min(intensities),
            'std_intensity': np.std(intensities),
            'trend_direction': self._calculate_trend(recent_analyses)
        }
    
    def _calculate_trend(self, analyses: list) -> str:
        """Calculate if intensity is trending up, down, or stable"""
        if len(analyses) < 3:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_analyses = sorted(analyses, key=lambda x: x.get('timestamp', datetime.min))
        
        # Compare first half to second half
        midpoint = len(sorted_analyses) // 2
        first_half_avg = np.mean([a['intensity_scale'] for a in sorted_analyses[:midpoint]])
        second_half_avg = np.mean([a['intensity_scale'] for a in sorted_analyses[midpoint:]])
        
        difference = second_half_avg - first_half_avg
        
        if difference > 5:
            return "increasing"
        elif difference < -5:
            return "decreasing"
        else:
            return "stable"
    
    def get_extremism_analysis(self) -> Dict[str, Any]:
        """Analyze extremism patterns in recent data"""
        if not self.analyses:
            return {}
        
        # Categories based on intensity
        low = [a for a in self.analyses if a['intensity_scale'] < 30]
        moderate = [a for a in self.analyses if 30 <= a['intensity_scale'] < 60]
        high = [a for a in self.analyses if 60 <= a['intensity_scale'] < 80]
        extreme = [a for a in self.analyses if a['intensity_scale'] >= 80]
        
        total = len(self.analyses)
        
        return {
            'total_analyses': total,
            'distribution': {
                'low_intensity': {'count': len(low), 'percentage': len(low)/total*100},
                'moderate_intensity': {'count': len(moderate), 'percentage': len(moderate)/total*100},
                'high_intensity': {'count': len(high), 'percentage': len(high)/total*100},
                'extreme_intensity': {'count': len(extreme), 'percentage': len(extreme)/total*100}
            },
            'most_extreme': max(self.analyses, key=lambda x: x['intensity_scale']) if self.analyses else None,
            'most_neutral': min(self.analyses, key=lambda x: x['intensity_scale']) if self.analyses else None
        }

class PoliticalKeywordExtractor:
    """Extract and analyze political keywords and themes"""
    
    def __init__(self):
        self.political_categories = {
            'healthcare': ['healthcare', 'health care', 'medicare', 'medicaid', 'obamacare', 'aca'],
            'economy': ['economy', 'economic', 'jobs', 'employment', 'inflation', 'gdp', 'market'],
            'taxation': ['tax', 'taxes', 'taxation', 'irs', 'revenue'],
            'immigration': ['immigration', 'immigrant', 'border', 'asylum', 'deportation'],
            'climate': ['climate', 'environment', 'carbon', 'renewable', 'fossil', 'green'],
            'education': ['education', 'school', 'teacher', 'student', 'university', 'college'],
            'defense': ['military', 'defense', 'army', 'navy', 'war', 'veteran'],
            'social_issues': ['abortion', 'gun', 'marriage', 'lgbt', 'religion', 'freedom'],
            'government': ['government', 'federal', 'state', 'congress', 'senate', 'house'],
            'election': ['election', 'vote', 'voting', 'ballot', 'campaign', 'democracy']
        }
        
        self.partisan_indicators = {
            'democratic': [
                'universal healthcare', 'climate action', 'progressive', 'social justice',
                'equality', 'inclusive', 'diversity', 'green new deal', 'medicare for all'
            ],
            'republican': [
                'fiscal responsibility', 'small government', 'traditional values', 'law and order',
                'border security', 'free market', 'constitutional rights', 'america first'
            ],
            'extreme_left': [
                'radical left', 'socialist', 'communist', 'antifa', 'defund police',
                'abolish', 'revolutionary', 'marxist'
            ],
            'extreme_right': [
                'radical right', 'fascist', 'nazi', 'supremacist', 'militia', 'insurrection',
                'deep state', 'stolen election'
            ]
        }
    
    def extract_themes(self, text: str) -> Dict[str, Any]:
        """Extract political themes from text"""
        text_lower = text.lower()
        
        themes = {}
        for category, keywords in self.political_categories.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                themes[category] = matches
        
        partisan_signals = {}
        for lean, keywords in self.partisan_indicators.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                partisan_signals[lean] = matches
        
        return {
            'themes': themes,
            'partisan_signals': partisan_signals,
            'theme_count': len(themes),
            'partisan_signal_count': len(partisan_signals)
        }
    
    def get_intensity_indicators(self, text: str) -> Dict[str, Any]:
        """Identify language patterns that indicate high intensity"""
        
        intensity_patterns = {
            'exclamation_marks': text.count('!'),
            'all_caps_words': len([word for word in text.split() if word.isupper() and len(word) > 2]),
            'extreme_adjectives': self._count_extreme_adjectives(text),
            'inflammatory_terms': self._count_inflammatory_terms(text),
            'absolute_statements': self._count_absolute_statements(text)
        }
        
        # Calculate intensity score based on patterns
        intensity_score = (
            intensity_patterns['exclamation_marks'] * 0.1 +
            intensity_patterns['all_caps_words'] * 0.2 +
            intensity_patterns['extreme_adjectives'] * 0.3 +
            intensity_patterns['inflammatory_terms'] * 0.4 +
            intensity_patterns['absolute_statements'] * 0.2
        )
        
        return {
            'patterns': intensity_patterns,
            'linguistic_intensity_score': min(intensity_score, 10),  # Cap at 10
            'intensity_level': self._categorize_intensity(intensity_score)
        }
    
    def _count_extreme_adjectives(self, text: str) -> int:
        """Count extreme/emotional adjectives"""
        extreme_adjectives = [
            'devastating', 'catastrophic', 'outrageous', 'disgraceful', 'radical',
            'extreme', 'dangerous', 'terrible', 'awful', 'brilliant', 'amazing',
            'incredible', 'unbelievable', 'fantastic', 'tremendous'
        ]
        text_lower = text.lower()
        return sum(1 for adj in extreme_adjectives if adj in text_lower)
    
    def _count_inflammatory_terms(self, text: str) -> int:
        """Count inflammatory political terms"""
        inflammatory_terms = [
            'destroy', 'enemy', 'traitor', 'corrupt', 'lies', 'fraud', 'scam',
            'betrayal', 'attack', 'war', 'fight', 'battle', 'crusade'
        ]
        text_lower = text.lower()
        return sum(1 for term in inflammatory_terms if term in text_lower)
    
    def _count_absolute_statements(self, text: str) -> int:
        """Count absolute statement indicators"""
        absolute_indicators = [
            'always', 'never', 'all', 'none', 'every', 'completely', 'totally',
            'absolutely', 'definitely', 'certainly', 'must', 'will'
        ]
        text_lower = text.lower()
        return sum(1 for indicator in absolute_indicators if indicator in text_lower)
    
    def _categorize_intensity(self, score: float) -> str:
        """Categorize linguistic intensity level"""
        if score >= 3:
            return "very_high"
        elif score >= 2:
            return "high"
        elif score >= 1:
            return "moderate"
        else:
            return "low"