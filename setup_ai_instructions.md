# Setting Up AI-Powered Clue Generation

## Overview

The `ai_semantic_rank.py` script can use OpenAI's GPT models to generate unique, contextual clues for each word in relation to your secret word. This creates much more engaging and accurate clues than pattern-based templates.

## Setup Instructions

### 1. Get an OpenAI API Key

1. Go to [https://platform.openai.com/](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (it starts with `sk-...`)

### 2. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**Permanent Setup (Windows):**
1. Search for "Environment Variables" in Start menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables..." button
4. Under "User variables", click "New..."
5. Variable name: `OPENAI_API_KEY`
6. Variable value: `sk-your-api-key-here`

### 3. Run with AI

```bash
python ai_semantic_rank.py
```

## AI vs Pattern-Based Clues

### With AI (OpenAI API key set):
- **earth** (rank 2): "Rocky neighbor of *"
- **telescope** (rank 500): "Tool to observe *"
- **book** (rank 5000): "Unlike * in space"
- **spoon** (rank 80000): "* are nothing like spoon"

### Without AI (fallback patterns):
- **earth** (rank 2): "A type of *"
- **telescope** (rank 500): "Related to *"
- **book** (rank 5000): "Like *"
- **spoon** (rank 80000): "* are nothing like spoon"

## Cost Considerations

- **Model**: GPT-3.5-turbo (cost-effective)
- **Tokens per clue**: ~20 tokens average
- **Total for 172k words**: ~3.4M tokens
- **Estimated cost**: $3-7 USD for full generation
- **Rate limits**: Built-in delays to respect API limits

## Features

- **Individual clues**: Each word gets a unique, contextual clue
- **Intelligent fallback**: Uses patterns when AI fails
- **Rate limit handling**: Automatic delays between requests
- **Error recovery**: Continues generation even if some AI calls fail
- **Progress tracking**: Shows real-time progress during generation

## Testing

Run the demo to see clue comparisons:
```bash
python demo_ai_clues.py
```

This shows side-by-side examples of AI vs pattern-based clues for the same words.
