# 🗳️ Political Tweet Intensity Analyzer

A Streamlit application that analyzes political sentiment and intensity in tweets using a fine-tuned DistilBERT model. Track how current political discourse compares to 2021 senator baseline data.

![Political Tweet Analyzer](https://img.shields.io/badge/Political-Tweet%20Analyzer-blue) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **🤖 AI-Powered Analysis**: Uses `m-newhauser/distilbert-political-tweets` model (90.76% accuracy)
- **📊 Comprehensive Analytics**: Track intensity trends, political breakdown, and temporal patterns
- **⚖️ Baseline Comparison**: Compare current discourse to 2021 senator tweet baseline
- **📈 Interactive Visualizations**: Rich charts and graphs using Plotly
- **💾 Data Persistence**: Automatic storage and analysis of tweet history
- **🔒 Privacy Focused**: Local data storage with anonymization options
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for model downloads)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/political-tweet-analyzer.git
cd political-tweet-analyzer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

That's it! The model will download automatically on first use.

### Alternative: One-Command Setup

Use the included run script:
```bash
python run.py
```

## 📊 How It Works

### The Model

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuned on**: 99,693 tweets from US senators during 2021
- **Classification**: Binary (Democratic vs Republican sentiment)
- **Performance**: 90.76% accuracy on evaluation set

### Analysis Metrics

1. **Political Lean**: Democratic or Republican based on language patterns
2. **Intensity Scale (0-100)**: How partisan/extreme the language is
3. **Confidence Score**: Model's confidence in the classification
4. **Baseline Comparison**: How the content compares to 2021 senator tweets

### Intensity Levels

- **0-30**: Neutral/mild political language
- **30-60**: Moderate partisan content
- **60-80**: Strong partisan language  
- **80-100**: Extremely partisan/intense language

## 🎮 Usage Guide

### Analyzing Tweets

1. **Navigate to "🔍 Analyze Tweets"**
2. **Enter tweet text** in the text area
3. **Use quick examples** or enter your own content
4. **Click "🔍 Analyze Tweet"**
5. **View comprehensive results** including:
   - Political lean and confidence
   - Intensity scale and baseline comparison
   - Visual gauges and charts

### Dashboard Analytics

1. **Go to "📊 Analytics Dashboard"**
2. **View comprehensive analytics**:
   - Intensity distribution charts
   - Political breakdown analysis
   - Timeline trends
   - Extreme content analysis
   - Temporal patterns

### Historical Trends

1. **Visit "📈 Historical Trends"**
2. **Analyze patterns over time**:
   - Daily intensity trends
   - Political direction changes
   - Period comparisons

## 📁 Project Structure

```
political-tweet-analyzer/
├── app.py                      # Main Streamlit application
├── political_analyzer.py       # DistilBERT model interface
├── tweet_tracker.py           # Data storage and analytics
├── analytics_dashboard.py     # Visualization components
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── run.py                    # Easy launch script
├── .gitignore               # Git ignore patterns
└── tweet_data/              # Data storage directory (auto-created)
    ├── analyzed_tweets.csv  # Tweet analysis results
    ├── analytics_cache.json # Cached analytics
    └── settings.json        # Application settings
```

## 🔧 Configuration

The app creates a `tweet_data/settings.json` file with these options:

```json
{
  "data_retention_days": 365,
  "anonymize_after_days": 30,
  "cache_duration_minutes": 15,
  "max_tweets_display": 100,
  "privacy_mode": false,
  "export_format": "csv"
}
```

## 📊 Data Export

Export your analysis data in multiple formats:

1. **CSV Export**: Raw tweet analysis data
2. **JSON Export**: Complete analytics with metadata
3. **Dashboard Export**: Comprehensive analytics report

Access via the "📥 Export Data" button in the Dashboard Controls.

## 🔒 Privacy & Data Handling

- **Local Storage**: All data stored locally on your machine
- **No External Tracking**: No data sent to third parties
- **Anonymization**: Session IDs automatically anonymized after 30 days
- **Data Retention**: Configurable retention periods
- **GDPR Compliant**: Full control over your data

## 🛠️ Advanced Usage

### Custom Model Integration

To use a different model, modify `political_analyzer.py`:

```python
self.model_name = "your-model-name-here"
```

### Batch Processing

Use the batch analysis feature:

```python
from political_analyzer import PoliticalAnalyzer

analyzer = PoliticalAnalyzer()
tweets = ["Tweet 1", "Tweet 2", "Tweet 3"]
results = analyzer.analyze_batch(tweets)
```

### API Integration

The core analyzer can be used as a standalone library:

```python
from political_analyzer import PoliticalAnalyzer

analyzer = PoliticalAnalyzer()
result = analyzer.analyze_tweet("Your political text here")
print(f"Political lean: {result['political_lean']}")
print(f"Intensity: {result['intensity_scale']}")
```

## 🔬 Research Applications

This tool is designed for:

- **Political Science Research**: Analyze discourse evolution
- **Media Analysis**: Study bias and polarization in news/social media
- **Academic Studies**: Research political communication patterns
- **Journalism**: Fact-check and analyze political statements
- **Education**: Demonstrate NLP and political science concepts

## ⚠️ Limitations & Considerations

- **Training Bias**: Model trained only on 2021 senator tweets
- **Context Limitation**: Optimized for short text (tweet-length)
- **Binary Classification**: Only distinguishes Democratic/Republican
- **Temporal Bias**: May not capture evolving political language
- **Not a Truth Detector**: Analyzes style, not factual accuracy

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional model integrations
- Enhanced visualization features
- Better temporal analysis
- Multi-language support
- Real-time Twitter integration

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Model Creator**: [m-newhauser](https://huggingface.co/m-newhauser) for the DistilBERT political tweets model
- **Hugging Face**: For the Transformers library and model hosting
- **Streamlit**: For the excellent web app framework
- **Plotly**: For interactive visualizations

## 📚 References

- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Model on Hugging Face](https://huggingface.co/m-newhauser/distilbert-political-tweets)
- [Senator Tweet Dataset](https://huggingface.co/datasets/m-newhauser/senator-tweets)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🐛 Issues & Support

- **Bug Reports**: Create an issue on GitHub
- **Feature Requests**: Open a discussion or issue
- **Questions**: Check existing issues or create a new one

## 🔄 Version History

- **v1.0.0**: Initial release with basic analysis features
- **v1.1.0**: Added comprehensive analytics dashboard
- **v1.2.0**: Enhanced visualizations and trend analysis
- **v1.3.0**: Improved privacy controls and data export

---

**Built with ❤️ for political science research and education**