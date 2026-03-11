# simplemodel-270M Training

his repository contains the training pipeline used to train simplemodel-270M, a decoder-only Transformer language model based on the GPT-NeoX architecture.

The repository includes scripts for data preprocessing, model training, and generate

# Model Architecture

Layers: 16

Hidden size: 1024

Feed-forward size: 4096

Attention heads: 8

Context length: 1024

Vocabulary size: 32000

Parameters: ~270M

# Installation
## Clone the repository and install dependencies.

git clone https://github.com/YOUR_USERNAME/simplemodel-training.git
cd simplemodel-training

pip install -r requirements.txt

### Example dependencies:
torch
sentencepiece
safetensors
transformers
vllm
datasets
pandas
numpy
tqdm

# Data
The dataset is processed through the following pipeline:
    raw_data -> cleaned_data(this is for training for LLM) -> 
    sentences_data(this is for training tokenizer) -> tokenizer.model

Run the following scripts to prepare the training dataset.
```bash
python Data/Scripts/data_loading.py
python Data/Scripts/Normalizing_cleaning.py
python Data/Scripts/tokenizer.py
```

# Train
The model follows a GPT-NeoX style decoder-only Transformer architecture.

Main configuration:
- Architecture: GPT-NeoX (decoder-only Transformer)
- Number of layers: 16
- Hidden size: 1024
- Feed-forward size: 4096
- Attention heads: 8
- Head dimension: 128
- Context length: 1024
- Vocabulary size: 32,000
- Total parameters: ~270M

Run the training script:

```bash
python train.py
```

# Generate
test the model by generate.py

```bash
python generate.py
```
