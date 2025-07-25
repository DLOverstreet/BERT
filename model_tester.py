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
        
        # Test cases with known expected results - expanded with more examples
        self.test_cases = {
            "clearly_democratic": [
                "Healthcare is a human right and we must ensure universal coverage for all Americans.",
                "We need to expand social safety nets and support working families.",
                "Climate change is real and we must take bold action to protect our planet.",
                "Everyone deserves equal rights regardless of their background.",
                "We should raise the minimum wage to help workers afford basic necessities.",
                "Healthcare should be a right for everyone",
                "We need universal healthcare coverage",
                "Tax the wealthy to fund social programs",
                "Expand Medicare for all Americans",
                "Fight climate change with government action"
            ],
            "clearly_republican": [
                "We need to cut taxes and reduce government spending to boost economic growth.",
                "Strong border security is essential for national sovereignty.",
                "We must protect the Second Amendment and gun rights.",
                "Free market capitalism creates prosperity and opportunity.",
                "We need to support law enforcement and maintain law and order.",
                "Cut taxes on businesses and families",
                "Secure our borders and enforce immigration law",
                "Protect our constitutional rights and freedoms",
                "Reduce government regulation on businesses",
                "Support our police and military"
            ],
            "senator_style_democratic": [
                "I believe healthcare is a fundamental right that every American deserves.",
                "We must invest in clean energy to combat climate change and create jobs.",
                "Working families need our support through expanded childcare programs.",
                "I will fight to raise the federal minimum wage to $15 per hour.",
                "Access to quality education should not depend on zip code."
            ],
            "senator_style_republican": [
                "I believe in limited government and fiscal responsibility.",
                "We must secure our border and enforce our immigration laws.",
                "I will always defend the Second Amendment and constitutional rights.",
                "Free enterprise and lower taxes create jobs and opportunity.",
                "I support our law enforcement officers and military heroes."
            ],
            "neutral": [
                "Education is important for our children's future.",
                "Infrastructure improvements benefit everyone.",
                "We should work together to solve problems.",
                "Technology is changing how we live and work.",
                "The weather has been nice this week."
            ]
        }
    
    def run_extreme_diagnostic(self):
        """Run extreme diagnostic testing to understand model behavior"""
        if not self.model_loaded:
            st.error("Model not loaded. Please load model first.")
            return
        
        st.write("## 🔬 Extreme Diagnostic Testing")
        st.write("Testing with very obvious examples to understand model behavior...")
        
        # Test extremely obvious examples
        extreme_tests = [
            ("I am a proud socialist who wants to redistribute wealth", "Should be Democratic"),
            ("I am a proud capitalist who wants to cut all taxes", "Should be Republican"),
            ("Socialism is the best economic system", "Should be Democratic"),
            ("Capitalism is the best economic system", "Should be Republican"),
            ("I love big government and high taxes", "Should be Democratic"),
            ("I love small government and low taxes", "Should be Republican"),
            ("Democrats are always right about everything", "Should be Democratic"),
            ("Republicans are always right about everything", "Should be Republican"),
            ("I hate Republicans and love Democrats", "Should be Democratic"),
            ("I hate Democrats and love Republicans", "Should be Republican")
        ]
        
        results = []
        
        for text, expected in extreme_tests:
            model_results = self.classifier(text)
            if isinstance(model_results, list) and len(model_results) > 0:
                if isinstance(model_results[0], list):
                    result_list = model_results[0]
                else:
                    result_list = model_results
                
                winner = max(result_list, key=lambda x: x['score'])
                predicted = winner['label']
                confidence = winner['score'] * 100
                
                expected_label = "Democratic" if "Democratic" in expected else "Republican"
                correct = expected_label.lower() in predicted.lower()
                
                results.append({
                    'text': text,
                    'expected': expected_label,
                    'predicted': predicted,
                    'confidence': f"{confidence:.1f}%",
                    'correct': "✅" if correct else "❌"
                })
        
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r['correct'] == "✅")
        total_count = len(results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        st.metric("Extreme Example Accuracy", f"{accuracy:.1f}%")
        
        if accuracy < 50:
            st.error("🚨 **MODEL IS FUNDAMENTALLY BROKEN** - Can't classify obvious examples")
            st.write("**Possible causes:**")
            st.write("- Training data was corrupted or mislabeled")
            st.write("- Model was trained for a different task")
            st.write("- Model has severe bias issues")
            st.write("- Wrong model version or corrupted weights")
            
            if st.button("🔍 Try Different Model"):
                st.code("# Try a different political classification model")
                st.write("Consider using cardiffnlp/twitter-roberta-base-sentiment-latest instead")
    
    def analyze_training_bias(self):
        """Analyze potential training bias patterns"""
        st.write("## 🎯 Training Bias Analysis")
        
        # Test different phrasings of the same concept
        concept_tests = {
            "Healthcare as right": [
                "Healthcare is a human right",
                "Healthcare should be a right for everyone", 
                "Access to healthcare is a fundamental right",
                "Every person deserves healthcare",
                "Healthcare access is a basic human right"
            ],
            "Tax reduction": [
                "We should cut taxes",
                "Reduce taxes on families",
                "Lower tax rates for everyone",
                "Tax cuts boost the economy",
                "Decrease government taxation"
            ],
            "Climate action": [
                "We must fight climate change",
                "Climate action is urgent",
                "Stop global warming now",
                "Environmental protection is crucial",
                "Green energy is the future"
            ]
        }
        
        for concept, phrases in concept_tests.items():
            st.write(f"### {concept}")
            
            predictions = []
            for phrase in phrases:
                results = self.classifier(phrase)
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        result_list = results[0]
                    else:
                        result_list = results
                    
                    winner = max(result_list, key=lambda x: x['score'])
                    predictions.append({
                        'phrase': phrase,
                        'predicted': winner['label'],
                        'confidence': winner['score'] * 100
                    })
            
            # Show consistency
            pred_labels = [p['predicted'] for p in predictions]
            unique_labels = set(pred_labels)
            
            if len(unique_labels) == 1:
                st.success(f"✅ Consistent: All phrases classified as {list(unique_labels)[0]}")
            else:
                st.warning(f"⚠️ Inconsistent: {len(unique_labels)} different classifications")
            
            # Show details
            for pred in predictions:
                color = "🔵" if "Democrat" in pred['predicted'] else "🔴"
                st.write(f"{color} **{pred['phrase']}** → {pred['predicted']} ({pred['confidence']:.1f}%)")
            
            st.write("---")
    
    def test_senator_language(self):
        """Test with actual senator-style language since that's what it was trained on"""
        st.write("## 🏛️ Senator Language Testing")
        st.write("Testing with formal political language similar to actual senator tweets...")
        
        senator_tests = {
            "Democratic Senator Style": [
                "Today I voted to expand healthcare access for working families across our great state.",
                "We must invest in clean energy jobs and infrastructure for our future.",
                "I'm proud to support legislation that raises the minimum wage for hardworking Americans.",
                "Climate change is real and we need bold action to protect our planet.",
                "Every child deserves access to quality, affordable education."
            ],
            "Republican Senator Style": [
                "Today I voted to reduce the tax burden on hardworking families and small businesses.",
                "We must secure our border and enforce our immigration laws to keep America safe.",
                "I'm proud to defend our Second Amendment rights and constitutional freedoms.",
                "Free market policies create jobs and opportunity for all Americans.",
                "We need to support our law enforcement heroes and military veterans."
            ]
        }
        
        for style, texts in senator_tests.items():
            st.write(f"### {style}")
            
            correct_predictions = 0
            total_predictions = 0
            
            for text in texts:
                results = self.classifier(text)
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        result_list = results[0]
                    else:
                        result_list = results
                    
                    winner = max(result_list, key=lambda x: x['score'])
                    predicted = winner['label']
                    confidence = winner['score'] * 100
                    
                    expected = "Democratic" if "Democratic" in style else "Republican"
                    correct = expected.lower() in predicted.lower()
                    
                    if correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    status = "✅" if correct else "❌"
                    color = "🔵" if "Democrat" in predicted else "🔴"
                    
                    st.write(f"{status} {color} **{predicted}** ({confidence:.1f}%) - {text[:60]}...")
            
            accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            st.metric(f"{style} Accuracy", f"{accuracy:.1f}%")
            st.write("---")
    
    def check_preprocessing_issues(self):
        """Check if there are preprocessing or tokenization issues"""
        st.write("## 🔧 Preprocessing Analysis")
        
        test_text = "Healthcare should be a right for everyone"
        
        # Show tokenization
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(test_text)
            token_ids = self.tokenizer.encode(test_text)
            
            st.write(f"**Original text:** {test_text}")
            st.write(f"**Tokens:** {tokens}")
            st.write(f"**Token IDs:** {token_ids}")
            st.write(f"**Token count:** {len(tokens)}")
            
            # Check for special tokens
            special_tokens = self.tokenizer.all_special_tokens
            st.write(f"**Special tokens:** {special_tokens}")
            
            # Test different preprocessing
            variations = [
                test_text.lower(),
                test_text.upper(), 
                test_text.replace("should", "must"),
                test_text.replace("everyone", "all Americans"),
                f"[CLS] {test_text} [SEP]"
            ]
            
            st.write("### Preprocessing Variations")
            for i, variation in enumerate(variations):
                results = self.classifier(variation)
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        result_list = results[0]
                    else:
                        result_list = results
                    
                    winner = max(result_list, key=lambda x: x['score'])
                    st.write(f"**Variation {i+1}:** {variation}")
                    st.write(f"→ {winner['label']} ({winner['score']*100:.1f}%)")
                    st.write("---")
    
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
        
        # Check for label reversal
        self._check_label_reversal(all_results)
    
    def _check_label_reversal(self, all_results):
        """Check if the model labels are reversed"""
        st.write("## 🔄 Label Reversal Analysis")
        
        st.write("""
        **Hypothesis**: The model's labels might be reversed. Let's check if swapping 
        "Republican" ↔ "Democratic" gives better results.
        """)
        
        # Test with obvious examples
        obvious_tests = {
            "Should be Democratic": [
                "Healthcare is a human right for everyone",
                "We need universal healthcare coverage", 
                "Climate change requires government action",
                "Expand social safety nets for working families"
            ],
            "Should be Republican": [
                "Cut taxes and reduce government spending",
                "Strong border security is essential",
                "Protect Second Amendment rights",
                "Free market capitalism creates prosperity"
            ]
        }
        
        total_correct_normal = 0
        total_correct_swapped = 0
        total_tests = 0
        
        results_table = []
        
        for expected_category, texts in obvious_tests.items():
            expected_label = "democrat" if "democratic" in expected_category.lower() else "republican"
            
            for text in texts:
                # Get model prediction
                model_results = self.classifier(text)
                if isinstance(model_results, list) and len(model_results) > 0:
                    if isinstance(model_results[0], list):
                        result_list = model_results[0]
                    else:
                        result_list = model_results
                    
                    winner = max(result_list, key=lambda x: x['score'])
                    predicted_label = winner['label'].lower()
                    confidence = winner['score'] * 100
                    
                    # Check normal interpretation
                    normal_correct = expected_label in predicted_label
                    
                    # Check swapped interpretation  
                    swapped_prediction = "democrat" if "republican" in predicted_label else "republican"
                    swapped_correct = expected_label in swapped_prediction
                    
                    results_table.append({
                        'text': text[:50] + "..." if len(text) > 50 else text,
                        'expected': expected_label.title(),
                        'model_says': winner['label'],
                        'confidence': f"{confidence:.1f}%",
                        'normal_correct': "✅" if normal_correct else "❌",
                        'swapped_correct': "✅" if swapped_correct else "❌"
                    })
                    
                    if normal_correct:
                        total_correct_normal += 1
                    if swapped_correct:
                        total_correct_swapped += 1
                    total_tests += 1
        
        # Display results table
        if results_table:
            st.write("### Test Results")
            
            import pandas as pd
            df = pd.DataFrame(results_table)
            st.dataframe(df, use_container_width=True)
            
            # Summary
            normal_accuracy = (total_correct_normal / total_tests * 100) if total_tests > 0 else 0
            swapped_accuracy = (total_correct_swapped / total_tests * 100) if total_tests > 0 else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Normal Interpretation Accuracy", f"{normal_accuracy:.1f}%")
            with col2:
                st.metric("Swapped Labels Accuracy", f"{swapped_accuracy:.1f}%")
            
            st.write("### 🔍 Diagnosis")
            
            if swapped_accuracy > normal_accuracy + 20:
                st.error("🚨 **LABELS ARE LIKELY REVERSED!**")
                st.write("The model performs much better when we swap Republican ↔ Democratic labels.")
                st.write("**Recommendation**: Implement label swapping in the analyzer.")
                
                # Show fix button
                if st.button("🔧 Apply Label Swap Fix"):
                    st.code("""
# Add this to political_analyzer.py in the _parse_model_results method:

# LABEL SWAP FIX - Model labels appear to be reversed
if any(keyword in label for keyword in ['DEMOCRATIC', 'DEMOCRAT', 'DEM']):
    rep_score = score  # Swap: Democratic label actually means Republican
elif any(keyword in label for keyword in ['REPUBLICAN', 'GOP', 'REP']):
    dem_score = score  # Swap: Republican label actually means Democratic
                    """)
                    st.info("Copy this code fix and apply it to your political_analyzer.py file")
            
            elif normal_accuracy < 30:
                st.error("🚨 **MODEL IS PERFORMING VERY POORLY**")
                st.write("Both normal and swapped interpretations have low accuracy.")
                st.write("**Possible issues**:")
                st.write("- Model was trained incorrectly")
                st.write("- Wrong model version")
                st.write("- Model has systematic bias")
                st.write("- Training data was mislabeled")
            
            elif normal_accuracy > 70:
                st.success("✅ **MODEL IS WORKING CORRECTLY**")
                st.write("Normal label interpretation shows good accuracy.")
            
            else:
                st.warning("⚠️ **MODEL HAS MODERATE ACCURACY**")
                st.write("Results are mixed. The model may have some bias or training issues.")
        
        # Additional debugging info
        st.write("### 🔬 Additional Debugging")
        
        if st.button("🔍 Test Label Mapping"):
            self._test_label_mapping()
    
    def _test_label_mapping(self):
        """Test the model's label mapping with extreme examples"""
        st.write("#### Extreme Example Testing")
        
        extreme_tests = [
            ("I love socialism and hate capitalism", "Should be Democratic"),
            ("I love capitalism and hate socialism", "Should be Republican"),
            ("Defund the police completely", "Should be Democratic"),
            ("Blue lives matter - support police", "Should be Republican"),
            ("Tax the rich at 90% rates", "Should be Democratic"), 
            ("Eliminate all taxes on businesses", "Should be Republican")
        ]
        
        for text, expected in extreme_tests:
            st.write(f"**Test**: {text}")
            st.write(f"**Expected**: {expected}")
            
            results = self.classifier(text)
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    result_list = results[0]
                else:
                    result_list = results
                
                for result in result_list:
                    label = result['label']
                    score = result['score'] * 100
                    st.write(f"- **{label}**: {score:.1f}%")
            
            st.write("---")
    
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