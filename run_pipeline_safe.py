#!/usr/bin/env python3
"""
Unicode-safe launcher for the CSV generation pipeline
Sets proper encoding environment and handles Unicode gracefully
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_unicode_environment():
    """Setup environment variables for proper Unicode handling"""
    env = os.environ.copy()
    
    # Python Unicode settings
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    
    # Windows console settings
    if os.name == 'nt':  # Windows
        env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    return env

def run_pipeline_with_unicode_safety():
    """Run the pipeline with enhanced Unicode safety"""
    print("🚀 Starting Unicode-Safe CSV Generation Pipeline")
    print("=" * 60)
    
    try:
        # Setup environment
        env = setup_unicode_environment()
        
        # Set console to UTF-8 on Windows
        if os.name == 'nt':
            try:
                subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
                print("✅ Console set to UTF-8 (CP65001)")
            except:
                print("⚠️ Could not set console to UTF-8, continuing anyway")
        
        # Run the main pipeline
        pipeline_script = Path("010_orchestrate_csv_pipeline.py")
        if not pipeline_script.exists():
            print("❌ Pipeline script not found: 010_orchestrate_csv_pipeline.py")
            return False
        
        print(f"🔄 Executing: {pipeline_script}")
        print("=" * 60)
        
        # Run with enhanced environment
        result = subprocess.run([
            sys.executable, str(pipeline_script)
        ], env=env, encoding='utf-8', errors='replace')
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ Pipeline completed successfully!")
            return True
        else:
            print(f"❌ Pipeline failed with exit code: {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_pipeline_with_unicode_safety()
    sys.exit(0 if success else 1)

