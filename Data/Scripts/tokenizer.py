import pandas as pd
import numpy as np
import os
import sentencepiece as spm
import re
from tqdm import tqdm

# Process English text
# sentence_count = 0
# with open("../raw_data/cleaned_eng.txt", "r", encoding="utf-8") as infile, \
#      open("../raw_data/sentences_eng.txt", "w", encoding="utf-8") as outfile:
#     for line_num, line in tqdm(enumerate(infile, 1), desc="Processing English", unit=" lines"):
#         sentences = re.split(r'[\.\n]+', line)
#         for sent in sentences:
#             sent = sent.strip()
#             if sent:
#                 outfile.write(sent + "\n")
#                 sentence_count += 1
#         # if line_num % 1000000 == 0:
#         #     print(f"  English: {line_num:,} 줄 처리됨, {sentence_count:,} 문장 저장됨")

# print(f"✓ English: 총 {sentence_count:,} 문장 저장됨")

# # Process Korean text
# sentence_count = 0
# with open("../raw_data/cleaned_kor.txt", "r", encoding="utf-8") as infile, \
#      open("../raw_data/sentences_kor.txt", "w", encoding="utf-8") as outfile:
#     for line_num, line in tqdm(enumerate(infile, 1), desc="Processing Korean", unit=" lines"):
#         sentences = re.split(r'[\.\n]+', line)
#         for sent in sentences:
#             sent = sent.strip()
#             if sent:
#                 outfile.write(sent + "\n")
#                 sentence_count += 1
#         # if line_num % 1000000 == 0:
#         #     print(f"  Korean: {line_num:,} 줄 처리됨, {sentence_count:,} 문장 저장됨")

# print(f"✓ Korean: 총 {sentence_count:,} 문장 저장됨")

# Load sentences for analysis (sample only)
print("\n" + "=" * 60)
print("Loading sentence samples for analysis...")
print("=" * 60)

sentence_eng = []
with open("../raw_data/sentences_eng.txt", "r", encoding="utf-8") as f:
    for i, line in tqdm(enumerate(f), desc="Loading English samples", total=100000, unit=" sentences"):
        if i >= 100000:  # Sample first 100k sentences
            break
        sentence_eng.append(line.strip())

sentence_kor = []
with open("../raw_data/sentences_kor.txt", "r", encoding="utf-8") as f:
    for i, line in tqdm(enumerate(f), desc="Loading Korean samples", total=100000, unit=" sentences"):
        if i >= 100000:  # Sample first 100k sentences
            break
        sentence_kor.append(line.strip())

print(f"✓ Loaded {len(sentence_eng):,} English sample sentences")
print(f"✓ Loaded {len(sentence_kor):,} Korean sample sentences")

print(f"\nEnglish sentences: {len(sentence_eng)}")
print(f"Korean sentences: {len(sentence_kor)}")

spm.SentencePieceTrainer.train(
    input="../raw_data/sentences_eng.txt,../raw_data/sentences_kor.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="unigram",
    character_coverage=0.9995,
    input_sentence_size=10000000,
    shuffle_input_sentence=True,
    # Special tokens
    pad_id=3,
    pad_piece="<pad>",
    bos_id=1,
    bos_piece="<s>",
    eos_id=2,
    eos_piece="</s>",
    unk_id=0,
    unk_piece="<unk>"
)

# Load trained tokenizer
print("\n" + "=" * 60)
print("Loading trained tokenizer...")
print("=" * 60)

sp = spm.SentencePieceProcessor()
sp.Load("tokenizer.model")

# Test tokenization and calculate coverage
print("\n" + "=" * 60)
print("Analyzing Token Coverage and Distribution...")
print("=" * 60)

def analyze_tokenization(sentences, language_name, sample_size=10000):
    """Analyze token coverage and length distribution"""
    print(f"\n Analyzing {language_name} text...")
    
    # Sample sentences for analysis
    sample_texts = sentences[:sample_size]
    
    # Token statistics
    all_token_lengths = []
    covered_chars = 0
    total_chars = 0
    unknown_tokens = 0
    total_tokens = 0
    
    for sent in sample_texts:
        total_chars += len(sent)
        tokens = sp.EncodeAsPieces(sent)
        
        for token in tokens:
            total_tokens += 1
            token_len = len(token)
            all_token_lengths.append(token_len)
            
            # Check if token is unknown
            if token == '<unk>':
                unknown_tokens += 1
            else:
                covered_chars += len(token)
    
    # Calculate coverage
    coverage = (covered_chars / total_chars * 100) if total_chars > 0 else 0
    unk_ratio = (unknown_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"\n  ✓ Token Coverage: {coverage:.2f}%")
    print(f"  ✓ Unknown Token Ratio: {unk_ratio:.4f}%")
    print(f"  ✓ Total Tokens: {total_tokens:,}")
    print(f"  ✓ Vocabulary Size: {sp.vocab_size()}")
    
    # Token length distribution
    print(f"\n  Token Length Distribution:")
    token_length_dist = pd.Series(all_token_lengths).value_counts().sort_index()
    for length, count in token_length_dist.items():
        percentage = (count / len(all_token_lengths)) * 100
        bar = "█" * int(percentage / 2)
        print(f"    Length {length}: {count:6,} ({percentage:5.2f}%) {bar}")
    
    print(f"- Average Token Length: {np.mean(all_token_lengths):.2f}")
    print(f"- Max Token Length: {max(all_token_lengths)}")
    print(f"- Min Token Length: {min(all_token_lengths)}")

# Analyze both languages
analyze_tokenization(sentence_eng, "English", sample_size=100000)
analyze_tokenization(sentence_kor, "Korean", sample_size=100000)

# Show sample tokenization
print("\n" + "=" * 60)
print("Sample Tokenization Results")
print("=" * 60)

sample_eng = "The quick brown fox jumps over the lazy dog"
sample_kor = "빠른 갈색 여우가 게으른 개를 뛰어넘는다"

print(f"\n English Sample: {sample_eng}")
tokens_eng = sp.EncodeAsPieces(sample_eng)
print(f"  Tokens ({len(tokens_eng)}): {tokens_eng}")

print(f"\n Korean Sample: {sample_kor}")
tokens_kor = sp.EncodeAsPieces(sample_kor)
print(f"  Tokens ({len(tokens_kor)}): {tokens_kor}")
