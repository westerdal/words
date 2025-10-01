# Secret Words CSV Generation with Lock File Management

**Task:** Create CSV files for secret words that have embeddings files but are missing CSV files, with proper concurrency control using a lock file system.

## Step-by-Step Process:

### 1. Scan for Target Words:
- Find all `embeddings-[secretword].txt` files in the `secretword/` directory
- Extract the secret word from each filename (between "embeddings-" and ".txt")
- Check which words already have corresponding CSV files (`secretword-easy-animals-[word].csv`)
- Create a list of words that need CSV generation

### 2. Lock File Management & Garbage Collection:
- Check if `secretword\.lock-csv.lock` file exists (this is the global lock file location)
- If it doesn't exist, create it
- **CRITICAL GARBAGE COLLECTION STEP:** Read the lock file contents and parse each line format: `[word] [timestamp]`
- Calculate current time and identify any timestamps older than 1 hour
- **Delete all lines with timestamps over 1 hour old** - this is essential cleanup to prevent stale locks from blocking progress
- Write the cleaned lock file back to disk immediately after garbage collection

### 3. Word Reservation:
- **IMPORTANT: Reserve only ONE word at a time** to allow other AI processes to work on different words simultaneously
- Select the first available word from the list that needs CSV generation
- Check if it's still reserved in the cleaned lock file
- If not reserved, add a new line: `[word] [current_timestamp]`
- Use ISO format timestamp (e.g., `2025-09-20T15:30:45Z`)
- Write the updated lock file
- Proceed to generate CSV for this single word only

### 4. Complete Main Program Processing Steps (MUST execute in order):

#### 4.1. Load ENABLE Word List:
- Load all 114,495 words from `data/enable2.txt`
- Filter out plural words using comprehensive pluralization rules:
  - Words ending in "-s" (except "ss", "us", "is")
  - Words ending in "-es" (boxes, buses)  
  - Words ending in "-ies" (babies, parties)
  - Words ending in "-ves" (wives, leaves)
  - Irregular plurals: men, women, children, feet, teeth, mice, people
  - Same singular/plural: sheep, deer, fish, series, species
- Result: ~114,495 → filtered singular words list
- Verify secret word is in the filtered list and not detected as plural

#### 4.2. Load Embeddings and Compute Semantic Rankings:
- Load embeddings from `.env/embeddings2.json`
- Verify secret word exists in embeddings
- Get secret word embedding vector and normalize it
- For each ENABLE word:
  - Get its embedding vector (if available)
  - Normalize the embedding vector
  - Compute cosine similarity with secret word embedding
  - Assign similarity score (-1.0 if word not in embeddings)
- Sort all words by similarity (descending), then alphabetically for ties
- Create rankings dictionary: `{word: {'rank': int, 'similarity': float}}`
- Display top 10 most similar words for verification

#### 4.3. OpenAI Similar Words Processing:
- Run `scripts/utilities/020_expand_vocabulary.py [word]` to get additional similar words
- This performs two-pass expansion:
  - **Primary pass**: Get 300+ direct semantic associations from OpenAI GPT-4
  - **Synonym expansion**: Get 3-8 synonyms for each primary word
- Cache results in `secretword/openai-[word]-twopass.txt`
- Validate against ENABLE2 dataset and add new words if needed
- This step expands the vocabulary beyond the base ENABLE list

#### 4.4. Generate AI Clues (Ranks 1-10,000):
- For words ranked 1-10,000, generate AI clues using OpenAI
- **Special case**: Secret word (rank 1) gets clue "This is the *."
- For other words, use batch processing (50 words at a time):
  - Create relationship-focused prompts describing connection to secret word
  - Use format: "young offspring of that animal" (not "puppy is a young dog")
  - Avoid mentioning secret word directly - use "that animal/creature/thing"
  - Return JSON format for structured parsing
  - Handle rate limits with small delays between batches
- Fallback to "ERROR" clues if OpenAI fails
- Track successful clue generation count

### 5. CSV Generation:
- **ONLY after completing all preprocessing steps above**
- Run `scripts/processing/030_generate_final_csv.py [word]` for the reserved word
- This will:
  - Load the embeddings file for the word
  - Process all ranked words with AI clue generation
  - Create CSV with format: `rank,secret_word,word,clue,connection_strength`
  - Save to `secretword/secretword-easy-animals-[word].csv`
  - Include ~114,000+ rows with proper ranking and clues
  - Use NULL clues for ranks beyond the AI processing limit

### 6. Cleanup:
- After successfully generating the CSV file, remove the corresponding line from the lock file
- Write the updated lock file back to disk
- Verify the CSV file was created successfully before removing the lock entry

## Garbage Collection Details:
- **When:** Performed immediately upon accessing the lock file, before any reservations
- **What:** Remove any line where `current_time - timestamp > 1 hour`
- **Why:** Prevents abandoned locks from blocking other AI processes indefinitely
- **How:** Parse timestamp, compare with current time, filter out stale entries

## Error Handling:
- If lock file is corrupted, recreate it with current reservations only
- If CSV generation fails for a word, keep the lock entry and log the error
- If another process has reserved a word, skip it and move to the next
- Handle file I/O errors gracefully

## File Formats:
- **Lock file:** `secretword\.lock-csv.lock` - Plain text, one word per line with timestamp
- **CSV format:** Header row + data rows with rank, secret_word, word, clue, connection_strength columns
- **Timestamp format:** ISO 8601 format for consistency

## Concurrency Notes:
- Always perform garbage collection first before checking reservations
- Always read-modify-write the lock file atomically
- **Reserve only ONE word per execution** to maximize parallelism with other AI processes
- Respect other AI processes' reservations that are still fresh (< 1 hour)
- Clean up your own reservations promptly after completion

## Main Program Steps Summary:
The complete processing pipeline follows this order:
1. **Word List Processing**: Load and filter ENABLE words (~114K → filtered)
2. **Semantic Ranking**: Compute cosine similarities using embeddings
3. **OpenAI Expansion**: Two-pass vocabulary expansion with GPT-4
4. **AI Clue Generation**: Create relationship-focused clues for top 10K words
5. **CSV Creation**: Generate final CSV with all rankings and clues

## Critical Dependencies:
- `data/enable2.txt` - Base word list (114,495 words)
- `.env/embeddings2.json` - Pre-computed word embeddings
- OpenAI API key - For similar words expansion and clue generation
- `scripts/utilities/020_expand_vocabulary.py` - Vocabulary expansion
- `scripts/processing/030_generate_final_csv.py` - Final CSV generation

**Expected Output:** Successfully generated CSV file for ONE unreserved secret word, with complete preprocessing pipeline execution, proper lock file management, and automatic cleanup of stale locks. Run the process multiple times to handle additional words.
