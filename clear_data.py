#!/usr/bin/env python3
"""
🧹 Data Cleanup Script

This script clears existing tweet data to resolve datetime format issues.
Run this if you're getting datetime parsing errors.
"""

import os
from pathlib import Path
import shutil

def clear_tweet_data():
    """Clear all tweet data to start fresh"""
    data_dir = Path("tweet_data")
    
    if not data_dir.exists():
        print("✅ No data directory found - nothing to clear")
        return
    
    files_to_clear = [
        "analyzed_tweets.csv",
        "analytics_cache.json"
    ]
    
    cleared_files = []
    for filename in files_to_clear:
        file_path = data_dir / filename
        if file_path.exists():
            try:
                file_path.unlink()
                cleared_files.append(filename)
                print(f"🗑️ Cleared: {filename}")
            except Exception as e:
                print(f"❌ Error clearing {filename}: {e}")
        else:
            print(f"📄 Not found: {filename}")
    
    if cleared_files:
        print(f"\n✅ Successfully cleared {len(cleared_files)} files")
        print("🔄 Restart your Streamlit app to start with fresh data")
    else:
        print("\n💡 No data files found to clear")

def backup_data():
    """Backup existing data before clearing"""
    data_dir = Path("tweet_data")
    backup_dir = Path("tweet_data_backup")
    
    if not data_dir.exists():
        print("No data directory to backup")
        return
    
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        shutil.copytree(data_dir, backup_dir)
        print(f"📦 Data backed up to: {backup_dir}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Political Tweet Analyzer - Data Cleanup")
    print("=" * 50)
    
    choice = input("\n1. Clear data only\n2. Backup then clear\n3. Cancel\n\nChoose (1-3): ").strip()
    
    if choice == "1":
        print("\n🗑️ Clearing data...")
        clear_tweet_data()
    elif choice == "2":
        print("\n📦 Creating backup...")
        if backup_data():
            print("\n🗑️ Clearing data...")
            clear_tweet_data()
        else:
            print("❌ Backup failed - data not cleared")
    elif choice == "3":
        print("❌ Cancelled")
    else:
        print("❌ Invalid choice")
    
    print("\n👋 Done!")