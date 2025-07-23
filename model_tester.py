"""
🔬 Model Tester - Debug and validate the DistilBERT model

This module helps diagnose model performance and label issues.
"""

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """Test and debug the political classification model"""
    
    def __init__(self):
        self.model_name = "m-newhauser/distilbert-political-tweets"
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.model_loaded = False
        
        # Test cases with known expected results
        self.test_cases = {
            "clearly_democratic": [
                "Healthcare is a human right and we must ensure universal coverage for all Americans.",
                "We need to expand social safety nets and support working families.",
                "Climate change is real and we must take bold action to protect our planet.",
                "Everyone deserves equal rights regardless of their background.",
                "We should raise the minimum wage to help workers afford basic necessities."
            ],
            "clearly_republican": [
                "We need to cut taxes and reduce government spending to boost economic growth.",
                "Strong border security is essential for national sovereignty.",
                "We must protect the Second Amendment and gun rights.",
                "Free market capitalism creates prosperity and opportunity.",
                "We need to support law enforcement and maintain law and order."
            ],
            "neutral": [
                "Education is important for our children's future.",
                "Infrastructure improvements benefit everyone.",
                "We should work together to solve problems.",
                "Technology is changing how we live and work.",
                "The weather has been nice this week."
            ]
        }
    
    def load_model_detailed(self):
        """Load model with detailed debugging"""
        try:
            st.write("🔍 **Loading model components...**")
            
            # Load tokenizer
            with st.spinner("Loading tokenizer..."):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                st.success(f"✅ Tokenizer loaded: {self.tokenizer.__class__.__name__}")
            
            # Load model
            with st.spinner("Loading model..."):
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                st.success(f"✅ Model loaded: {self.model.__class__.__name__}")
                
                # Check model configuration
                config = self.model.config
                st.write(f"**Model Config:**")
                st.write(f"- Labels: {config.label2id if hasattr(config, 'label2id') else 'Not found'}")
                st.write(f"- Number of labels: {config.num_labels}")
                st.write(f"- Architecture: {config.architectures}")
            
            # Load pipeline
            with st.spinner("Loading pipeline..."):
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    top_k=None,
                    device=-1
                )
                st.success("✅ Pipeline loaded successfully")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def test_single_input(self, text):
        """Test a single input with detailed output"""
        if not self.model_loaded:
            return None
        
        try:
            st.write(f"🧪 **Testing:** '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Raw model prediction
            results = self.classifier(text)
            st.write(f"**Raw Results:** {results}")
            
            # Process results
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    result_list = results[0]  # Nested format
                else:
                    result_list = results  # Direct format
                
                for result in result_list:
                    label = result['label']
                    score = result['score']
                    confidence_pct = score * 100
                    
                    # Color code by political lean
                    if 'democrat' in label.lower():
                        st.markdown(f"🔵 **{label}**: {confidence_pct:.1f}%")
                    elif 'republican' in label.lower():
                        st.markdown(f"🔴 **{label}**: {confidence_pct:.1f}%")
                    else:
                        st.markdown(f"⚪ **{label}**: {confidence_pct:.1f}%")
                
                # Determine winner
                winner = max(result_list, key=lambda x: x['score'])
                st.write(f"**Classification:** {winner['label']} ({winner['score']*100:.1f}% confidence)")
                
                return result_list
            else:
                st.error(f"Unexpected result format: {results}")
                return None
                
        except Exception as e:
            st.error(f"❌ Test failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def run_test_suite(self):
        """Run comprehensive test suite"""
        if not self.model_loaded:
            st.error("Model not loaded. Please load model first.")
            return
        
        st.write("## 🧪 Comprehensive Test Suite")
        
        all_results = {}
        
        for category, test_texts in self.test_cases.items():
            st.write(f"### {category.replace('_', ' ').title()}")
            
            category_results = []
            for text in test_texts:
                with st.expander(f"Test: {text[:40]}..."):
                    result = self.test_single_input(text)
                    if result:
                        category_results.append({
                            'text': text,
                            'results': result
                        })
            
            all_results[category] = category_results
        
        # Summary analysis
        self._analyze_test_results(all_results)
    
    def _analyze_test_results(self, all_results):
        """Analyze test results for patterns"""
        st.write("## 📊 Test Analysis Summary")
        
        total_tests = sum(len(results) for results in all_results.values())
        st.write(f"**Total tests run:** {total_tests}")
        
        # Count correct classifications
        correct_count = 0
        total_count = 0
        
        for category, results in all_results.items():
            expected_label = None
            if 'democratic' in category:
                expected_label = 'democrat'
            elif 'republican' in category:
                expected_label = 'republican'
            
            if expected_label:
                for result_data in results:
                    if result_data['results']:
                        winner = max(result_data['results'], key=lambda x: x['score'])
                        actual_label = winner['label'].lower()
                        
                        if expected_label in actual_label:
                            correct_count += 1
                        total_count += 1
        
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            st.metric("Classification Accuracy", f"{accuracy:.1f}%")
            
            if accuracy < 70:
                st.warning("⚠️ Model accuracy is below 70%. This suggests:")
                st.write("- Model may be using unexpected label formats")
                st.write("- Training data may not match our test cases")
                st.write("- Model may need different preprocessing")
        
        # Label distribution
        st.write("### Label Distribution")
        label_counts = {}
        for results in all_results.values():
            for result_data in results:
                if result_data['results']:
                    winner = max(result_data['results'], key=lambda x: x['score'])
                    label = winner['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in sorted(label_counts.items()):
            st.write(f"- **{label}**: {count} predictions")
    
    def test_custom_input(self, text):
        """Test custom user input"""
        if not text or len(text.strip()) < 3:
            return None
        
        text = text.strip()
        st.write("### 🎯 Custom Input Test")
        
        return self.test_single_input(text)
    
    def show_model_info(self):
        """Display detailed model information"""
        st.write("## 🤖 Model Information")
        
        if not self.model_loaded:
            st.info("Load model first to see detailed information")
            return
        
        # Model configuration
        config = self.model.config
        st.write("### Configuration")
        config_dict = config.to_dict()
        
        important_keys = ['model_type', 'num_labels', 'label2id', 'id2label', 'vocab_size']
        for key in important_keys:
            if key in config_dict:
                st.write(f"- **{key}**: {config_dict[key]}")
        
        # Tokenizer info
        st.write("### Tokenizer")
        st.write(f"- **Type**: {self.tokenizer.__class__.__name__}")
        st.write(f"- **Vocab size**: {len(self.tokenizer)}")
        
        # Test tokenization
        test_text = "This is a test of the tokenization process"
        tokens = self.tokenizer.tokenize(test_text)
        st.write(f"- **Sample tokenization**: {test_text}")
        st.write(f"- **Tokens**: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")

def show_model_tester():
    """Main interface for model testing"""
    st.title("🔬 Model Testing & Debugging")
    st.markdown("*Comprehensive testing of the DistilBERT political classification model*")
    
    tester = ModelTester()
    
    # Model loading section
    st.write("## 1. Model Loading")
    if st.button("🔄 Load Model with Debugging"):
        success = tester.load_model_detailed()
        if success:
            st.balloons()
    
    # Model info
    if st.button("ℹ️ Show Model Information"):
        tester.show_model_info()
    
    # Test suite
    st.write("## 2. Automated Testing")
    if st.button("🧪 Run Full Test Suite"):
        tester.run_test_suite()
    
    # Custom testing
    st.write("## 3. Custom Testing")
    custom_text = st.text_area(
        "Enter your own text to test:",
        height=100,
        placeholder="Type any political statement to test the model..."
    )
    
    if st.button("🎯 Test Custom Input") and custom_text:
        tester.test_custom_input(custom_text)
    
    # Quick tests
    st.write("## 4. Quick Tests")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Democratic Statement"):
            result = tester.test_custom_input("Healthcare is a human right for all Americans")
    
    with col2:
        if st.button("Test Republican Statement"):
            result = tester.test_custom_input("We need to cut taxes and reduce government spending")

if __name__ == "__main__":
    show_model_tester()