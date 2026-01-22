import re
import os

source_file = r'c:\space\words\words\256-color-word-list.md'
target_file = r'c:\space\words\words\256-color-word-list.txt'

with open(source_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract the list content
# Looking for colors = [...]
match = re.search(r'colors\s*=\s*\[(.*?)\]', content, re.DOTALL)
if not match:
    print("Could not find colors list in file.")
    exit(1)

list_content = match.group(1)

# Extract strings (words between quotes)
# This handles "Word", "Word", etc.
words = re.findall(r'"([^"]+)"', list_content)

print(f"Found {len(words)} words.")

# Check for duplicates
seen = set()
duplicates = []
for w in words:
    if w in seen:
        duplicates.append(w)
    seen.add(w)

if duplicates:
    print(f"Warning: Found duplicates: {duplicates}")

print(f"Unique words: {len(seen)}")

# Write to file
with open(target_file, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print(f"Written to {target_file}")
