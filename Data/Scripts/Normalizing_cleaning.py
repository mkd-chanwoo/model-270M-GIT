import pandas as pd
import numpy as np
import os
import unicodedata
import re
import string

# Process and save English text line by line
print("=" * 60)
print("Processing English text (wikipedia_eng.txt)...")
print("=" * 60)

line_count = 0
cleaned_lines = 0

with open("wikipedia_eng.txt", "r", encoding="utf-8") as infile, \
     open("cleaned_eng.txt", "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line_count += 1
        
        # Normalize unicode
        line = unicodedata.normalize("NFC", line)
        
        # Apply cleaning operations
        line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
        line = re.sub(r'http\S+|www.\S+', '', line)  # Remove URLs
        line = re.sub(r'[^\w\s]', ' ', line)  # Remove punctuation/special chars
        line = re.sub(r'\s+', ' ', line)  # Remove extra whitespace
        line = line.strip()  # Strip leading/trailing spaces
        
        # Save non-empty lines
        if line:
            outfile.write(line + '\n')
            cleaned_lines += 1
        
        # Log progress every 10000 lines
        if line_count % 10000 == 0:
            print(f"  진행 상황: {line_count:,} 줄 읽음, {cleaned_lines:,} 줄 저장됨")

print(f"\n✓ 완료!")
print(f"  총 읽은 줄: {line_count:,}")
print(f"  총 저장된 줄: {cleaned_lines:,}")
print(f"  파일: cleaned_eng.txt")

# Process and save Korean text line by line
print("\n")
print("=" * 60)
print("Processing Korean text (korean_text.txt)...")
print("=" * 60)

line_count = 0
cleaned_lines = 0

with open("korean_text.txt", "r", encoding="utf-8") as infile, \
     open("cleaned_kor.txt", "w", encoding="utf-8") as outfile:
    
    for line in infile:
        line_count += 1
        
        # Normalize unicode
        line = unicodedata.normalize("NFC", line)
        
        # Apply cleaning operations
        line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
        line = re.sub(r'http\S+|www.\S+', '', line)  # Remove URLs
        line = re.sub(r'[^\w\s]', ' ', line)  # Remove punctuation/special chars
        line = re.sub(r'\s+', ' ', line)  # Remove extra whitespace
        line = line.strip()  # Strip leading/trailing spaces
        
        # Save non-empty lines
        if line:
            outfile.write(line + '\n')
            cleaned_lines += 1
        
        # Log progress every 10000 lines
        if line_count % 10000 == 0:
            print(f"  진행 상황: {line_count:,} 줄 읽음, {cleaned_lines:,} 줄 저장됨")

print(f"\n✓ 완료!")
print(f"  총 읽은 줄: {line_count:,}")
print(f"  총 저장된 줄: {cleaned_lines:,}")
print(f"  파일: cleaned_kor.txt")