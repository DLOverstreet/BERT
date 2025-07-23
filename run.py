#!/usr/bin/env python3
"""
🗳️ Political Tweet Intensity Analyzer - Easy Launch Script

This script will check dependencies and launch the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'plotly',
        'transformers',
        'torch',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("\n🔧 Installing missing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ])
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("   Please run manually: pip install -r requirements.txt")
            return False
    else:
        print("✅ All required dependencies found!")
        return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'app.py',
        'political_analyzer.py', 
        'tweet_tracker.py',
        'analytics_dashboard.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("   Please ensure all project files are present")
        return False
    
    print("✅ All required files found!")
    return True

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path("tweet_data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Created data directory")
    
    # Create .gitkeep file to preserve empty directory in git
    gitkeep_file = data_dir / ".gitkeep"
    if not gitkeep_file.exists():
        gitkeep_file.touch()

def launch_app():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Political Tweet Intensity Analyzer...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C in this terminal to stop the server")
    print("🌐 App will be available at: http://localhost:8501")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.headless', 'false',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Political Tweet Intensity Analyzer stopped")
        print("   Thanks for using our tool! 🗳️")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("🔧 Try running manually: streamlit run app.py")

def show_header():
    """Show application header"""
    print("=" * 60)
    print("🗳️  POLITICAL TWEET INTENSITY ANALYZER")
    print("=" * 60)
    print("📊 Analyze political sentiment using AI")
    print("🤖 Powered by DistilBERT")
    print("📈 Track discourse intensity trends")
    print("=" * 60)

def show_quick_help():
    """Show quick help information"""
    print("\n💡 QUICK HELP:")
    print("• Navigate to '🔍 Analyze Tweets' to analyze individual tweets")
    print("• Use '📊 Analytics Dashboard' to view comprehensive statistics") 
    print("• Check '📈 Historical Trends' for temporal analysis")
    print("• Read the 'ℹ️ About' section for model details")
    print("\n🔬 RESEARCH FEATURES:")
    print("• Export data for academic research")
    print("• Compare current discourse to 2021 baseline")
    print("• Track political intensity over time")
    print("• Analyze extremism patterns")

def main():
    """Main execution function"""
    show_header()
    
    # Pre-flight checks
    if not check_python_version():
        return
    
    if not check_files():
        return
    
    if not check_dependencies():
        return
    
    create_data_directory()
    
    print("\n🎯 All systems ready!")
    
    # Show help before launching
    show_quick_help()
    
    # Ask user if they want to continue
    print("\n" + "─" * 60)
    user_input = input("Press Enter to launch the app (or 'q' to quit): ").strip().lower()
    
    if user_input in ['q', 'quit', 'exit']:
        print("👋 Goodbye!")
        return
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main()