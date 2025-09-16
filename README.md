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
