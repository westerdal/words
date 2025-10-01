# Generate Missing AI Twopass Files

## Overview

This document explains how to generate `[secretword]-openai-twopass.txt` files for words that have CSV files but are missing their corresponding AI expansion data. These files contain the two-pass AI expansion results that significantly improve the quality and coverage of word associations.

**Note:** While this system currently uses OpenAI's GPT-3.5-turbo, the process can be adapted for other AI providers (Claude, Gemini, local models, etc.) by modifying the API calls and authentication methods.

## Current Status Analysis

Based on the current system state, the following words have CSV files but are **missing** OpenAI twopass files:
- `air` - Has air-secret.csv but missing air-openai-twopass.txt
- `art` - Has art-secret.csv but missing art-openai-twopass.txt  
- `bird` - Has bird-secret.csv but missing bird-openai-twopass.txt
- `cat` - Has cat-secret.csv but missing cat-openai-twopass.txt
- `dog` - Has dog-secret.csv but missing dog-openai-twopass.txt
- `fish` - Has fish-secret.csv but missing fish-openai-twopass.txt
- `forest` - Has forest-secret.csv but missing forest-openai-twopass.txt
- `horse` - Has horse-secret.csv but missing horse-openai-twopass.txt

*Note: cat_incomplete, dog_backup, dog_temp are likely temporary/backup files and can be ignored.*

## What Are AI Twopass Files?

AI twopass files contain the results of a sophisticated two-pass expansion process:

### **Pass 1: Primary Associations**
- Requests 300+ direct semantic associations from the AI model (currently OpenAI GPT-3.5-turbo)
- Uses carefully crafted prompts to get diverse, high-quality word associations
- Filters out generic words (thing, item, stuff, object, etc.)

### **Pass 2: Synonym Expansion** 
- Takes each primary word and generates 3-8 synonyms
- Expands the vocabulary significantly (typically 300 → 1,500+ words)
- Removes duplicates and validates against the ENABLE2 word list

### **AI Provider Flexibility**
The system is designed to work with different AI providers:
- **OpenAI**: GPT-3.5-turbo, GPT-4 (current implementation)
- **Anthropic**: Claude models (requires API modification)
- **Google**: Gemini models (requires API modification)
- **Local Models**: Ollama, LM Studio, etc. (requires endpoint modification)

### **File Format**
```
# Two-pass expansion for '[word]'
# Total words: 1497
# Generated: 2024-09-21T14:23:45.123456

word1, word2, word3, word4, word5, word6, ...
```

## Generation Methods

### Method 1: Individual Word Generation (Recommended)

**Command:**
```bash
python -c "
import sys
sys.path.append('scripts/utilities')
from openai_similar_words import OpenAISimilarWords

# Generate for specific word
word = 'TARGET_WORD_HERE'
print(f'Generating OpenAI twopass for: {word}')

try:
    module = OpenAISimilarWords(word)
    words = module.get_similar_words()
    
    if words:
        print(f'✅ SUCCESS: Generated {len(words)} words for {word}')
        print(f'File saved: {word}-openai-twopass.txt')
    else:
        print(f'❌ FAILED: No words generated for {word}')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
"
```

**Usage Example:**
```bash
# Replace TARGET_WORD_HERE with actual word
python -c "..." # (replace TARGET_WORD_HERE with 'air')
```

### Method 2: Batch Generation Script

**Create a batch script** `generate_missing_openai.py`:

```python
#!/usr/bin/env python3
"""
Generate missing OpenAI twopass files for words that have CSV files
but are missing their corresponding OpenAI expansion data.
"""

import sys
from pathlib import Path
sys.path.append('scripts/utilities')
from openai_similar_words import OpenAISimilarWords

# Words missing OpenAI twopass files (as of current analysis)
MISSING_WORDS = [
    'air', 'art', 'bird', 'cat', 'dog', 
    'fish', 'forest', 'horse'
]

def generate_missing_openai_files():
    """Generate OpenAI twopass files for missing words"""
    
    print(f"🚀 Starting batch generation for {len(MISSING_WORDS)} words")
    print(f"Words to process: {', '.join(MISSING_WORDS)}")
    
    success_count = 0
    failed_words = []
    
    for i, word in enumerate(MISSING_WORDS, 1):
        print(f"\n[{i}/{len(MISSING_WORDS)}] Processing: {word}")
        
        try:
            # Check if file already exists (skip if created since analysis)
            twopass_file = Path(f"secretword/{word}-openai-twopass.txt")
            if twopass_file.exists():
                print(f"⏭️  SKIP: {word}-openai-twopass.txt already exists")
                success_count += 1
                continue
            
            # Generate OpenAI expansion
            module = OpenAISimilarWords(word)
            words = module.get_similar_words()
            
            if words:
                print(f"✅ SUCCESS: Generated {len(words)} words for '{word}'")
                print(f"📁 Saved: {word}-openai-twopass.txt")
                success_count += 1
            else:
                print(f"❌ FAILED: No words generated for '{word}'")
                failed_words.append(word)
                
        except Exception as e:
            print(f"❌ ERROR processing '{word}': {e}")
            failed_words.append(word)
    
    # Summary
    print(f"\n🎯 BATCH GENERATION COMPLETE")
    print(f"✅ Successful: {success_count}/{len(MISSING_WORDS)}")
    
    if failed_words:
        print(f"❌ Failed: {len(failed_words)} words")
        print(f"Failed words: {', '.join(failed_words)}")
    else:
        print(f"🎉 All words processed successfully!")

if __name__ == "__main__":
    generate_missing_openai_files()
```

**Usage:**
```bash
python generate_missing_openai.py
```

### Method 3: Integration with csv-prompt

The `run_csv_prompt.py` script automatically generates OpenAI twopass files when processing words, so you could also:

1. **Temporarily remove the CSV files** for words missing twopass files
2. **Run csv-prompt** - it will regenerate both the OpenAI twopass and CSV files
3. **Advantage**: Ensures complete consistency between twopass and CSV data

## Prerequisites

### 1. AI API Configuration

**For OpenAI (Current Implementation):**
```bash
# Check if set
echo $env:OPENAI_API_KEY

# Set if missing (replace with your key)
$env:OPENAI_API_KEY = "sk-proj-..."
```

**For Other AI Providers:**
- **Claude**: Set `ANTHROPIC_API_KEY` environment variable
- **Gemini**: Set `GOOGLE_API_KEY` environment variable  
- **Local Models**: Configure endpoint URL in config files

### 2. Required Dependencies
```bash
# For OpenAI (current)
pip install openai

# For Claude
pip install anthropic

# For Gemini  
pip install google-generativeai

# For local models
pip install requests  # or specific client library
```

### 3. File Structure
Ensure these directories exist:
- `secretword/` - Where twopass files are saved
- `scripts/utilities/` - Contains the OpenAI processing scripts
- `data/enable2.txt` - Word validation list

## Expected Results

After successful generation, each word should have:

```
secretword/
├── [word]-secret.csv              # Main game data
├── [word]-openai-twopass.txt      # OpenAI expansion (NEW)
├── [word]-embeddings.txt          # Semantic embeddings
└── [word]-cache.json              # Clue cache
```

**File sizes to expect:**
- Small words (air, cat): ~800-1,200 words
- Medium words (bird, fish): ~1,200-1,800 words  
- Large words (forest, horse): ~1,500-2,500 words

## Troubleshooting

### Common Issues

**1. API Authentication Errors**
```
❌ ERROR: API key not found
❌ FATAL: OpenAI connection error
❌ FATAL: OpenAI API error
```
**Solutions:** 
- Set the appropriate environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.)
- Verify API key is valid and has sufficient credits
- Check network connectivity

**2. Rate Limiting**
```
❌ ERROR: Rate limit exceeded
❌ FATAL: OpenAI rate limit exceeded
```
**Solution:** 
- Wait 1-2 minutes between generations
- Use Method 2 with delays between requests
- Consider upgrading API plan for higher limits

**3. Connection Failures**
```
❌ FATAL: OpenAI connection failed
🛑 ABORTING: OpenAI connection failed
```
**Solution:** 
- Check internet connection
- Verify API endpoint is accessible
- System now **aborts immediately** on connection errors (no more infinite loops)

**4. Empty Results**
```
❌ FAILED: No words generated
```
**Solution:** Check word spelling, ensure it's in ENABLE2 list, verify API key permissions

**5. Unicode Errors**
```
UnicodeEncodeError: 'charmap' codec can't encode
```
**Solution:** Already fixed in the current codebase with fallback encoding

### Validation Commands

**Check if generation was successful:**
```bash
# Count words in generated file
Get-Content secretword/[word]-openai-twopass.txt | Select-String "^[^#]" | ForEach-Object { ($_ -split ',').Length }

# Verify file format
Get-Content secretword/[word]-openai-twopass.txt -Head 5
```

**Expected output:**
```
# Two-pass expansion for '[word]'
# Total words: 1234
# Generated: 2024-09-21T...

word1, word2, word3, word4, ...
```

## Integration Notes

Once AI twopass files are generated:

1. **csv-prompt will use them automatically** - No code changes needed
2. **Performance improvement** - Words with twopass files process faster (cached data)
3. **Quality improvement** - Enhanced word associations beyond base embeddings
4. **Consistency** - All words will have the same level of AI enhancement

## Adapting for Other AI Providers

To use a different AI provider, modify the following files:

### **scripts/utilities/openai_similar_words.py**
Replace the `_try_openai_request()` method:

```python
# Example for Claude
def _try_claude_request(self, prompt: str, max_tokens: int) -> List[str]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        content = response.content[0].text.strip()
        return self._parse_and_clean_response(content)
    except Exception as e:
        raise Exception(f"Claude API error: {e}")
```

### **scripts/utilities/config.py**
Add configuration for different providers:

```python
# AI Provider Settings
AI_PROVIDER = "openai"  # or "claude", "gemini", "local"
CLAUDE_MODEL = "claude-3-sonnet-20240229"
GEMINI_MODEL = "gemini-pro"
LOCAL_ENDPOINT = "http://localhost:11434/api/generate"  # for Ollama
```

## Recommended Workflow

1. **Start with Method 1** for 1-2 test words to verify everything works
2. **Use Method 2** for batch generation of all missing files  
3. **Verify results** using validation commands
4. **Run csv-prompt** to confirm integration works correctly
5. **Monitor performance** - should see faster processing for words with twopass files

---

## Error Handling Improvements

**Recent Updates:**
- ✅ **Connection errors now abort immediately** - No more infinite loops on API failures
- ✅ **Specific error types handled**: `APIConnectionError`, `APIError`, `RateLimitError`
- ✅ **Clear error messages** - Distinguishes between connection, authentication, and rate limit issues
- ✅ **Graceful shutdown** - System exits cleanly on fatal errors instead of continuing with empty results

**Before these improvements:** The system would silently fail API calls and continue processing with empty results, leading to incomplete data.

**After these improvements:** The system immediately aborts on connection errors, providing clear feedback about what went wrong.

---

*This document should be updated whenever new words are added to the system or when the AI expansion process is modified.*
