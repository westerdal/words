
import os

target_file = r'c:\space\words\words\256-color-word-list.txt'

with open(target_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Split by whitespace to handle spaces, tabs, newlines mixed
words = content.split()

print(f"Found {len(words)} words in text file.")

with open(target_file, 'w', encoding='utf-8') as f:
    for word in words:
        f.write(word + '\n')

print(f"Reformatted {target_file} to one word per line.")
