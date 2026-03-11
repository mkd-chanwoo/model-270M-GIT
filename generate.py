import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch
from safetensors.torch import load_file as safetensors_load_file

from model.model import model as GPTNeoXLikeModel


def load_model(checkpoint_path: Path, config, device: torch.device) -> GPTNeoXLikeModel:
    m = GPTNeoXLikeModel(config).to(device)
    state_dict = safetensors_load_file(str(checkpoint_path), device=str(device))
    m.load_state_dict(state_dict)
    m.eval()
    return m


@torch.no_grad()
def generate(
    model: GPTNeoXLikeModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eos_id: int,
) -> torch.Tensor:
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Truncate to max_position_embeddings if needed
        context = generated[:, -model.gpt_neox.embed_in.weight.shape[0]:]  # safety fallback
        context = generated

        logits = model(context)           # (1, seq_len, vocab_size)
        next_logits = logits[:, -1, :]    # (1, vocab_size)

        if temperature > 0.0:
            next_logits = next_logits / temperature

        if top_k > 0:
            values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = values[:, -1].unsqueeze(-1)
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

        if temperature > 0.0:
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        if next_token.item() == eos_id:
            break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text with model-200M")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_200m_last/model.safetensors")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--tokenizer", type=str, default="Data/tokenizer.model")
    parser.add_argument("--prompt", type=str, default="The cat never jump on the table because")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8, help="0.0 = greedy")
    parser.add_argument("--top-k", type=int, default=50, help="0 = disabled")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    with (root / args.config).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "max_position_embeddings" not in cfg:
        cfg["max_position_embeddings"] = cfg.get("initial_context_length", 1024)
    config = SimpleNamespace(**cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    if not sp.Load(str(root / args.tokenizer)):
        raise RuntimeError(f"Failed to load tokenizer: {args.tokenizer}")

    checkpoint_path = root / args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, config, device)

    token_ids = sp.encode(args.prompt, out_type=int)
    if config.bos_token_id >= 0:
        token_ids = [config.bos_token_id] + token_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=config.eos_token_id,
    )

    generated_tokens = output_ids[0, input_ids.size(1):].tolist()
    output_text = sp.decode(generated_tokens)
    print(args.prompt + output_text)


if __name__ == "__main__":
    main()
