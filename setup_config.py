#!/usr/bin/env python3
"""
Setup script for configuring the word game environment
"""
import os
import sys
from config import Config

def setup_openai_key():
    """Interactive setup for OpenAI API key"""
    print("🔧 OpenAI API Key Setup")
    print("=" * 40)
    
    current_key = Config.OPENAI_API_KEY
    if current_key and current_key != "your_openai_api_key_here":
        print(f"✅ Current API key: {current_key[:8]}...{current_key[-4:] if len(current_key) > 12 else current_key}")
        update = input("Update API key? (y/n): ").strip().lower()
        if update not in ['y', 'yes']:
            return
    
    print("\n💡 You can get your OpenAI API key from: https://platform.openai.com/api-keys")
    api_key = input("Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided!")
        return
    
    # Update config.env file
    config_lines = []
    key_updated = False
    
    if os.path.exists("config.env"):
        with open("config.env", 'r') as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    config_lines.append(f"OPENAI_API_KEY={api_key}\n")
                    key_updated = True
                else:
                    config_lines.append(line)
    
    if not key_updated:
        config_lines.append(f"OPENAI_API_KEY={api_key}\n")
    
    with open("config.env", 'w') as f:
        f.writelines(config_lines)
    
    # Set environment variable for current session
    os.environ['OPENAI_API_KEY'] = api_key
    Config.OPENAI_API_KEY = api_key
    
    print("✅ OpenAI API key configured successfully!")
    
    # Test the connection
    try:
        print("🔍 Testing OpenAI connection...")
        client = Config.get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        print("✅ OpenAI connection successful!")
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        print("💡 Please check your API key and try again.")

def show_config():
    """Display current configuration"""
    print("🔧 Current Configuration")
    print("=" * 40)
    print(f"OpenAI API Key: {'✅ Set' if Config.validate_openai_key() else '❌ Not set'}")
    print(f"Max Hints: {Config.MAX_HINTS}")
    print(f"Data Directory: {Config.DATA_DIR}")
    print(f"Secretword Directory: {Config.SECRETWORD_DIR}")
    print(f"Dictionary File: {Config.DICTIONARY_FILE}")
    print(f"Embeddings File: {Config.EMBEDDINGS_FILE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Save Interval: {Config.SAVE_INTERVAL}")
    print(f"Consecutive Weak Threshold: {Config.CONSECUTIVE_WEAK_THRESHOLD}")

def main():
    """Main setup menu"""
    while True:
        print("\n🎮 Word Game Configuration Setup")
        print("=" * 40)
        print("1. Setup OpenAI API Key")
        print("2. Show Current Configuration") 
        print("3. Test OpenAI Connection")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            setup_openai_key()
        elif choice == '2':
            show_config()
        elif choice == '3':
            if Config.validate_openai_key():
                try:
                    print("🔍 Testing OpenAI connection...")
                    client = Config.get_openai_client()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello, this is a test."}],
                        max_tokens=10
                    )
                    print("✅ OpenAI connection successful!")
                    print(f"Response: {response.choices[0].message.content}")
                except Exception as e:
                    print(f"❌ OpenAI connection failed: {e}")
            else:
                print("❌ OpenAI API key not configured!")
        elif choice == '4':
            print("👋 Setup complete!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()

