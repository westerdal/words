#!/usr/bin/env python3
"""
Test OpenAI API connection and capture detailed error information
"""

import openai
import os
import sys
from datetime import datetime

def test_openai_connection():
    """Test OpenAI connection and provide detailed error information"""
    
    print("=" * 60)
    print("🔧 OPENAI API CONNECTION DIAGNOSTIC")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    print(f"📦 OpenAI Library Version: {openai.__version__}")
    
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    print(f"🔑 API Key Found: {api_key[:12]}...{api_key[-8:]} (masked for security)")
    print(f"🔑 API Key Length: {len(api_key)} characters")
    
    # Test the connection with a simple request
    print("\n🧪 Testing API Connection...")
    print("-" * 40)
    
    try:
        # Try a simple completion request
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test message."}],
            max_tokens=10,
            temperature=0
        )
        
        print("✅ SUCCESS: OpenAI API connection working!")
        print(f"📝 Response: {response.choices[0].message.content}")
        
    except openai.AuthenticationError as e:
        print("❌ AUTHENTICATION ERROR:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print(f"   Error Code: {getattr(e, 'code', 'N/A')}")
        print("\n💡 POSSIBLE SOLUTIONS:")
        print("   1. Check if API key is correct")
        print("   2. Verify API key hasn't expired")
        print("   3. Check OpenAI account billing status")
        print("   4. Generate a new API key at https://platform.openai.com/account/api-keys")
        
    except openai.RateLimitError as e:
        print("❌ RATE LIMIT ERROR:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print("\n💡 POSSIBLE SOLUTIONS:")
        print("   1. Wait a moment and try again")
        print("   2. Check your usage limits")
        print("   3. Upgrade your OpenAI plan if needed")
        
    except openai.APIError as e:
        print("❌ API ERROR:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print(f"   Status Code: {getattr(e, 'status_code', 'N/A')}")
        print("\n💡 POSSIBLE SOLUTIONS:")
        print("   1. Check OpenAI service status")
        print("   2. Try again in a few minutes")
        print("   3. Contact OpenAI support if persistent")
        
    except Exception as e:
        print("❌ UNEXPECTED ERROR:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print(f"   Full Error Details: {repr(e)}")
        
        # Try to get more details
        if hasattr(e, 'response'):
            print(f"   HTTP Response: {e.response}")
        if hasattr(e, 'status_code'):
            print(f"   Status Code: {e.status_code}")
            
    print("\n" + "=" * 60)
    print("📋 INFORMATION FOR YOUR TEAM:")
    print("=" * 60)
    print("Please share this diagnostic output with your team.")
    print("They can help resolve the API key issue based on the error details above.")
    print("\n🔗 Useful Links:")
    print("   • OpenAI API Keys: https://platform.openai.com/account/api-keys")
    print("   • OpenAI Billing: https://platform.openai.com/account/billing")
    print("   • OpenAI Status: https://status.openai.com/")
    print("   • OpenAI Documentation: https://platform.openai.com/docs")

if __name__ == "__main__":
    test_openai_connection()


