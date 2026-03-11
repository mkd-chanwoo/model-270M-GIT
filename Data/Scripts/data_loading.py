from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import hashlib

ds = load_dataset("NeuML/wikipedia-20250123")
ds

wiki = pd.DataFrame(ds['train'])

# take only the first half of the rows to reduce size

half = wiki.iloc[: len(wiki)//2]
half = half.iloc[: len(half)//2]

half["hash"] = half["text"].apply(
    lambda x: hashlib.md5(x.encode()).hexdigest()
)
half = half.drop_duplicates(subset="hash")
print(half.head())

# write concatenated text to file without loading all into memory
# use the half-size dataframe
if 'text' in half.columns:
    with open('wikipedia_eng.txt', 'w', encoding='utf-8') as f:
        for i, text in enumerate(half['text']):
            f.write(str(text))
            if i < len(half) - 1:
                f.write('\n\n')
    print(f"wrote {len(half)} texts to wikipedia_eng.txt")
else:
    print("No 'text' column found in half dataframe")


if os.path.exists('wikipedia_eng.txt'):
    size_bytes = os.path.getsize('wikipedia_eng.txt')
    size_gb = size_bytes / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GB")
else:
    print("wikipedia_eng.txt not found")




# korean 2

ds_kor2 = load_dataset("HAERAE-HUB/KOREAN-WEBTEXT")

text_kor2 = pd.DataFrame(ds_kor2['train'])

text_kor2 = text_kor2.iloc[: len(text_kor2)//2]
text_kor2 = text_kor2.iloc[: len(text_kor2)//2]

text_kor2["hash"] = text_kor2["text"].apply(
    lambda x: hashlib.md5(x.encode()).hexdigest()
)
text_kor2 = text_kor2.drop_duplicates(subset="hash")
print(text_kor2.head())

# write concatenated text to file without loading all into memory
# use the half-size dataframe
if 'text' in text_kor2.columns:
    with open('korean_text2.txt', 'w', encoding='utf-8') as f:
        for i, text in enumerate(text_kor2['text']):
            f.write(str(text))
            if i < len(text_kor2) - 1:
                f.write('\n\n')
    print(f"wrote {len(text_kor2)} texts to korean_text2.txt")
else:
    print("No 'text' column found in half dataframe")

if os.path.exists('korean_text2.txt'):
    size_bytes = os.path.getsize('korean_text2.txt')
    size_gb = size_bytes / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GB")
else:
    print("korean_text2.txt not found")