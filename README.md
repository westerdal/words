# Python Project

A new Python project ready for development.

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd words
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

## Development

### Adding Dependencies
Add new packages to `requirements.txt` and run:
```bash
pip install -r requirements.txt
```

### Deactivating Virtual Environment
When you're done working:
```bash
deactivate
```

## Project Structure
```
words/
├── data/
│   └── enable1.txt     # ENABLE word list (172,823 words)
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── venv/               # Virtual environment (created after setup)
```

## Word List

This project includes the **ENABLE word list** (Enhanced North American Benchmark Lexicon), which contains 172,823 English words commonly used in word games and applications. The word list is stored in `data/enable1.txt` and is in the public domain.

### Using the Word List

The `main.py` script includes a `load_word_list()` function that reads and processes the word list:

```python
from main import load_word_list

# Load all words
words = load_word_list()
print(f"Loaded {len(words)} words")

# Find words of specific length
five_letter_words = [word for word in words if len(word) == 5]
```

## Semantic Rank Game

This project includes a **Semantic Rank Game CSV Generator** that creates precomputed data files for word guessing games based on semantic similarity.

### Features

- **Complete ENABLE Word Coverage**: Ranks all 172,823 words relative to a secret word
- **AI-Powered Clue Generation**: Uses OpenAI GPT to write unique clues for each word
- **Intelligent Fallback**: Pattern-based clues when AI is unavailable
- **Instant Gameplay**: No real-time computation needed during gameplay
- **Customizable Secret Words**: Easy to generate files for different secret words

### AI vs Pattern-Based Clues

**With AI (requires OpenAI API key):**
- **earth** → "Rocky neighbor of *"
- **telescope** → "Tool to observe *"
- **book** → "Unlike * in space"

**Without AI (fallback patterns):**
- **earth** → "Similar to *"
- **telescope** → "Related to *"
- **book** → "Like *"

### Usage

**Generate with AI clues (recommended):**
```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="sk-your-key-here"
python ai_semantic_rank.py
```

**Generate with pattern-based clues:**
```bash
python simple_semantic_rank.py
```

**Compare clue types:**
```bash
python demo_ai_clues.py
```

**Analyze generated data:**
```bash
python analyze_csv.py
```

### Setup AI Clue Generation

See `setup_ai_instructions.md` for detailed instructions on:
- Getting an OpenAI API key
- Setting environment variables
- Cost considerations (~$3-7 for full generation)

### Game Data Structure

The CSV follows the exact specification:
- **Rank 1**: Secret word ("This is the *.")
- **Ranks 2-1000**: Closest words (AI-generated or semantic relationships)
- **Ranks 1001-5000**: Medium similarity (contextual clues)
- **Ranks 5001-50000**: Weak associations
- **Ranks 50001+**: Distant words ("* are nothing like [word].")

### Generated Files

- `ai_semantic_rank_planet.csv` - AI-powered clues (if API key available)
- `semantic_rank_planet.csv` - Pattern-based clues
- Both contain: rank, secret_word, word, clue columns
