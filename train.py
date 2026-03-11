import argparse
import atexit
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import sentencepiece as spm
import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model.model import model as GPTNeoXLikeModel


class TokenChunkDataset(Dataset):
	def __init__(self, token_ids: List[int], seq_len: int):
		if len(token_ids) < seq_len + 1:
			raise ValueError("Not enough tokens to build training samples.")
		self.token_ids = token_ids
		self.seq_len = seq_len

	def __len__(self) -> int:
		return len(self.token_ids) - self.seq_len

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		x = self.token_ids[idx : idx + self.seq_len]
		y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
		return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_model_config(config_path: Path) -> SimpleNamespace:
	with config_path.open("r", encoding="utf-8") as f:
		cfg = json.load(f)

	if "max_position_embeddings" not in cfg:
		cfg["max_position_embeddings"] = cfg.get("initial_context_length", 1024)

	return SimpleNamespace(**cfg)


def stream_tokenize_files(
	sp: spm.SentencePieceProcessor,
	text_paths: List[Path],
	max_tokens: int,
) -> List[int]:
	token_ids: List[int] = []
	bos_id = sp.bos_id()
	eos_id = sp.eos_id()

	for path in text_paths:
		if not path.exists():
			raise FileNotFoundError(f"Training text file not found: {path}")

		with path.open("r", encoding="utf-8") as f:
			for line in f:
				text = line.strip()
				if not text:
					continue

				if bos_id >= 0:
					token_ids.append(bos_id)
				token_ids.extend(sp.encode(text, out_type=int))
				if eos_id >= 0:
					token_ids.append(eos_id)

				if max_tokens > 0 and len(token_ids) >= max_tokens:
					return token_ids[:max_tokens]

	return token_ids


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
	def lr_lambda(current_step: int) -> float:
		if current_step < warmup_steps:
			return float(current_step + 1) / float(max(1, warmup_steps))

		progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		progress = min(max(progress, 0.0), 1.0)
		cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
		return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def normalize_checkpoint_dir(path: Path) -> Path:
	# If user passes a file-like path (e.g., *.pt), use its stem as checkpoint directory.
	if path.suffix:
		return path.with_suffix("")
	return path


def progress_bar(percent: float, width: int = 30) -> str:
	percent = min(max(percent, 0.0), 100.0)
	filled = int(round((percent / 100.0) * width))
	return "#" * filled + "-" * (width - filled)


class TeeStream:
	"""Writes to both console and log file."""

	def __init__(self, primary_stream, log_stream):
		self.primary_stream = primary_stream
		self.log_stream = log_stream

	def write(self, data):
		try:
			self.primary_stream.write(data)
		except Exception:
			pass
		try:
			self.log_stream.write(data)
			self.log_stream.flush()
		except Exception:
			pass
		return len(data) if data else 0

	def flush(self):
		try:
			self.primary_stream.flush()
		except Exception:
			pass
		try:
			self.log_stream.flush()
		except Exception:
			pass

	def isatty(self):
		return False

	def fileno(self):
		raise io.UnsupportedOperation("fileno")


def enable_file_logging(log_path: Path):
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_file = log_path.open("w", encoding="utf-8", buffering=1)

	original_stdout = sys.stdout
	original_stderr = sys.stderr
	sys.stdout = TeeStream(original_stdout, log_file)
	sys.stderr = TeeStream(original_stderr, log_file)

	def _cleanup():
		try:
			if hasattr(sys.stdout, "flush"):
				sys.stdout.flush()
			if hasattr(sys.stderr, "flush"):
				sys.stderr.flush()
		except Exception:
			pass
		finally:
			sys.stdout = original_stdout
			sys.stderr = original_stderr
			try:
				log_file.close()
			except Exception:
				pass

	atexit.register(_cleanup)


def save_checkpoint(
	checkpoint_path: Path,
	model: nn.Module,
	optimizer,
	scheduler,
	scaler,
	epoch: int,
	global_step: int,
	best_val_loss: float,
):
	checkpoint_path = normalize_checkpoint_dir(checkpoint_path)
	checkpoint_path.mkdir(parents=True, exist_ok=True)
	model_path = checkpoint_path / "model.safetensors"
	state_path = checkpoint_path / "training_state.pt"

	# Save full model weights in safetensors for safe and portable checkpointing.
	state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
	safetensors_save_file(state_dict_cpu, str(model_path))

	payload = {
		"optimizer": optimizer.state_dict(),
		"scheduler": scheduler.state_dict(),
		"scaler": scaler.state_dict() if scaler is not None else None,
		"epoch": epoch,
		"global_step": global_step,
		"best_val_loss": best_val_loss,
	}
	torch.save(payload, str(state_path))


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer, scheduler, scaler, device: torch.device):
	# Legacy format support: single-file checkpoint (*.pt)
	if checkpoint_path.is_file():
		payload = torch.load(str(checkpoint_path), map_location=device)
		if "model" not in payload:
			raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
		model.load_state_dict(payload["model"])
		optimizer.load_state_dict(payload["optimizer"])
		scheduler.load_state_dict(payload["scheduler"])
		if scaler is not None and payload.get("scaler") is not None:
			scaler.load_state_dict(payload["scaler"])
		return (
			int(payload.get("epoch", 0)),
			int(payload.get("global_step", 0)),
			float(payload.get("best_val_loss", float("inf"))),
		)

	checkpoint_path = normalize_checkpoint_dir(checkpoint_path)
	# New format: checkpoint directory with model.safetensors + training_state.pt
	if checkpoint_path.is_dir():
		model_path = checkpoint_path / "model.safetensors"
		state_path = checkpoint_path / "training_state.pt"
		if not model_path.exists() or not state_path.exists():
			raise FileNotFoundError(f"Invalid checkpoint dir: {checkpoint_path}")

		model_state = safetensors_load_file(str(model_path), device="cpu")
		model.load_state_dict(model_state)

		payload = torch.load(str(state_path), map_location=device)
		optimizer.load_state_dict(payload["optimizer"])
		scheduler.load_state_dict(payload["scheduler"])
		if scaler is not None and payload.get("scaler") is not None:
			scaler.load_state_dict(payload["scaler"])
	else:
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	return (
		int(payload.get("epoch", 0)),
		int(payload.get("global_step", 0)),
		float(payload.get("best_val_loss", float("inf"))),
	)


def query_nvidia_smi() -> str:
	try:
		cmd = [
			"nvidia-smi",
			"--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
			"--format=csv,noheader,nounits",
		]
		out = subprocess.check_output(cmd, text=True, timeout=2).strip().splitlines()
		if not out:
			return "nvidia-smi: no GPU info"
		util, mem_used, mem_total, temp = [x.strip() for x in out[0].split(",")]
		return f"GPU={util}% VRAM={mem_used}/{mem_total}MB Temp={temp}C"
	except Exception:
		return "nvidia-smi unavailable"


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	amp_enabled: bool,
	micro_batch_size: int,
) -> float:
	model.eval()
	losses: List[float] = []

	for input_ids_cpu, labels_cpu in loader:
		batch_size = input_ids_cpu.size(0)
		chunk_size = batch_size if micro_batch_size <= 0 else min(micro_batch_size, batch_size)
		batch_loss = 0.0

		for start in range(0, batch_size, chunk_size):
			end = start + chunk_size
			input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)
			labels = labels_cpu[start:end].to(device, non_blocking=True)

			with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
				logits = model(input_ids)
				loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

			weight = input_ids.size(0) / max(1, batch_size)
			batch_loss += loss.item() * weight

		losses.append(batch_loss)

	model.train()
	return float(sum(losses) / max(1, len(losses)))


def main():
	parser = argparse.ArgumentParser(description="Train model-200M with CE + AdamW + warmup/decay")
	parser.add_argument("--config", type=str, default="config.json")
	parser.add_argument("--tokenizer", type=str, default="Data/tokenizer.model")
	parser.add_argument(
		"--train-files",
		nargs="+",
		default=["Data/train_data/cleaned_eng.txt", "Data/train_data/cleaned_kor.txt"],
	)
	parser.add_argument("--seq-len", type=int, default=1024)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=0.1)
	parser.add_argument("--warmup-steps", type=int, default=200)
	parser.add_argument("--min-lr-ratio", type=float, default=0.1)
	parser.add_argument("--max-steps", type=int, default=0, help="0 means no explicit max steps")
	parser.add_argument("--max-train-tokens", type=int, default=5000, help="0 means no token cap")
	parser.add_argument("--val-ratio", type=float, default=0.01)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--grad-accum-steps", type=int, default=10)
	parser.add_argument("--save-every", type=int, default=500)
	parser.add_argument("--eval-every", type=int, default=200)
	parser.add_argument("--log-every", type=int, default=100)
	parser.add_argument("--savepoint", type=str, default="checkpoints/model_200m_last")
	parser.add_argument("--log-file", type=str, default="", help="Log path relative to project root. Default: <savepoint>/train.log")
	parser.add_argument("--micro-batch-size", type=int, default=1, help="Split each loader batch into smaller chunks to reduce CUDA peak memory. 0 uses full batch.")
	parser.add_argument("--resume", type=str, default="")
	parser.add_argument("--no-amp", action="store_true")
	args = parser.parse_args()

	if args.grad_accum_steps < 1:
		raise ValueError("--grad-accum-steps must be >= 1")
	if args.micro_batch_size < 0:
		raise ValueError("--micro-batch-size must be >= 0")
	if not (0.0 < args.val_ratio < 0.5):
		raise ValueError("--val-ratio must be in (0, 0.5)")

	root = Path(__file__).resolve().parent
	savepoint_path = normalize_checkpoint_dir(root / args.savepoint)
	log_path = (root / args.log_file) if args.log_file else (root / "log" / "train.log")
	enable_file_logging(log_path)
	print(f"Logging to: {log_path}")

	config = load_model_config(root / args.config)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	amp_enabled = False
	amp_dtype = torch.float16
	if torch.cuda.is_available() and not args.no_amp:
		if torch.cuda.is_bf16_supported():
			amp_enabled = True
			amp_dtype = torch.bfloat16
			print("AMP enabled with bf16")
		else:
			amp_enabled = True
			amp_dtype = torch.float16
			print("AMP enabled with fp16")

	scaler = torch.amp.GradScaler("cuda", enabled=(torch.cuda.is_available() and amp_dtype == torch.float16 and amp_enabled))

	sp = spm.SentencePieceProcessor()
	tokenizer_path = root / args.tokenizer
	if not sp.Load(str(tokenizer_path)):
		raise RuntimeError(f"Failed to load tokenizer: {tokenizer_path}")

	text_paths = [root / p for p in args.train_files]
	token_ids = stream_tokenize_files(sp, text_paths, max_tokens=args.max_train_tokens)
	if len(token_ids) < args.seq_len + 2:
		raise ValueError("Tokenized data is too small for selected sequence length.")

	split_idx = int(len(token_ids) * (1.0 - args.val_ratio))
	train_ids = token_ids[:split_idx]
	val_ids = token_ids[split_idx - args.seq_len - 1 :]

	print(f"Total tokens: {len(token_ids):,} | train={len(train_ids):,} | val={len(val_ids):,}")

	train_ds = TokenChunkDataset(train_ids, seq_len=args.seq_len)
	val_ds = TokenChunkDataset(val_ids, seq_len=args.seq_len)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		drop_last=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	model = GPTNeoXLikeModel(config).to(device)
	criterion = nn.CrossEntropyLoss(ignore_index=getattr(config, "pad_token_id", -100))
	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
	total_steps = steps_per_epoch * args.epochs
	if args.max_steps > 0:
		total_steps = min(total_steps, args.max_steps)

	scheduler = build_scheduler(
		optimizer,
		warmup_steps=args.warmup_steps,
		total_steps=max(1, total_steps),
		min_lr_ratio=args.min_lr_ratio,
	)

	start_epoch = 0
	global_step = 0
	best_val_loss = float("inf")

	if args.resume:
		resume_path = root / args.resume
		if not resume_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
		start_epoch, global_step, best_val_loss = load_checkpoint(
			resume_path, model, optimizer, scheduler, scaler, device
		)
		print(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")

	savepoint_path = root / args.savepoint
	model.train()
	optimizer.zero_grad(set_to_none=True)
	stop_training = False
	last_logged_percent = -1

	for epoch in range(start_epoch, args.epochs):
		running_train_loss = 0.0
		micro_step = 0

		for step_in_epoch, (input_ids_cpu, labels_cpu) in enumerate(train_loader, start=1):
			batch_size = input_ids_cpu.size(0)
			chunk_size = batch_size if args.micro_batch_size <= 0 else min(args.micro_batch_size, batch_size)
			batch_loss = 0.0
			oom_in_batch = False

			for start in range(0, batch_size, chunk_size):
				end = start + chunk_size
				input_ids = input_ids_cpu[start:end].to(device, non_blocking=True)
				labels = labels_cpu[start:end].to(device, non_blocking=True)

				try:
					with torch.amp.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype):
						logits = model(input_ids)
						loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
				except RuntimeError as e:
					if "out of memory" in str(e).lower():
						print(
							"CUDA OOM detected in micro-batch. "
							"Try smaller --micro-batch-size (e.g., 1), --seq-len, or --batch-size."
						)
						oom_in_batch = True
						if torch.cuda.is_available():
							torch.cuda.empty_cache()
						break
					raise

				weight = input_ids.size(0) / max(1, batch_size)
				batch_loss += loss.item() * weight
				scaled_loss = (loss * weight) / args.grad_accum_steps
				scaler.scale(scaled_loss).backward()

			if oom_in_batch:
				optimizer.zero_grad(set_to_none=True)
				continue

			running_train_loss += batch_loss

			micro_step += 1
			should_step = (micro_step % args.grad_accum_steps == 0) or (step_in_epoch == len(train_loader))
			if not should_step:
				continue

			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			prev_scale = scaler.get_scale()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)

			if scaler.get_scale() >= prev_scale:
				scheduler.step()

			global_step += 1
			total_progress = (100.0 * global_step) / max(1, total_steps)
			epoch_progress = (100.0 * step_in_epoch) / max(1, len(train_loader))
			progress_tick = int(total_progress)
			should_log_progress_tick = progress_tick > last_logged_percent

			if global_step % args.log_every == 0 or global_step == 1 or should_log_progress_tick:
				if should_log_progress_tick:
					last_logged_percent = progress_tick
				current_lr = scheduler.get_last_lr()[0]
				avg_train_loss = running_train_loss / max(1, step_in_epoch)
				bar = progress_bar(total_progress)
				msg = (
					f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
					f"total=[{bar}] {total_progress:6.2f}% "
					f"epoch_progress={epoch_progress:6.2f}% "
					f"lr={current_lr:.8f} train_loss={avg_train_loss:.6f}"
				)
				if torch.cuda.is_available():
					alloc = torch.cuda.memory_allocated() / (1024 ** 3)
					reserved = torch.cuda.memory_reserved() / (1024 ** 3)
					msg += f" | cuda_mem={alloc:.2f}G alloc / {reserved:.2f}G reserved"
					msg += f" | {query_nvidia_smi()}"
				print(msg)

			if global_step % args.eval_every == 0:
				val_loss = evaluate(model, val_loader, criterion, device, amp_enabled, args.micro_batch_size)
				print(f"epoch={epoch + 1} step={global_step} val_loss={val_loss:.6f}")
				if val_loss < best_val_loss:
					best_val_loss = val_loss
					best_path = savepoint_path.with_name(savepoint_path.name + "_best")
					save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss)
					print(f"Best checkpoint updated: {best_path / 'model.safetensors'}")

			if global_step % args.save_every == 0:
				save_checkpoint(savepoint_path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss)
				print(f"Checkpoint saved: {savepoint_path / 'model.safetensors'}")

			if args.max_steps > 0 and global_step >= args.max_steps:
				stop_training = True
				break

		epoch_val_loss = evaluate(model, val_loader, criterion, device, amp_enabled, args.micro_batch_size)
		epoch_train_loss = running_train_loss / max(1, len(train_loader))
		print(
			f"epoch={epoch + 1} completed | train_loss={epoch_train_loss:.6f} "
			f"| val_loss={epoch_val_loss:.6f} | best_val_loss={best_val_loss:.6f}"
		)

		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss

		save_checkpoint(savepoint_path, model, optimizer, scheduler, scaler, epoch + 1, global_step, best_val_loss)
		print(f"Epoch checkpoint saved: {savepoint_path / 'model.safetensors'}")

		if stop_training:
			break

	print("Training complete.")


if __name__ == "__main__":
	main()
