#!/usr/bin/env python3
# train_splice_single_task_kmer_bert.py
#
# Train a SMALL BERT-like (encoder-only Transformer) with k-mer tokenization (default k=3)
# to score splice sites as probabilities in [0, 1].
#
# IMPORTANT (your data convention):
# - The "positive" file (.pwm.err) contains REAL splice sites => label = 1
# - The "negative" file (.neg.pwm.err) contains decoy sites      => label = 0
# - The trailing + / - in each line is STRAND and is ignored.
#
# This script trains a SINGLE-TASK model:
#   --task donor     trains only on donor sequences
#   --task acceptor  trains only on acceptor sequences
#
# Usage:
#   python train_splice_single_task_kmer_bert.py \
#       --task donor \
#       --pos /path/to/GCF_...fna.pwm.err \
#       --neg /path/to/GCF_...fna.neg.pwm.err \
#       --out_dir donor_model \
#       --k 3 --max_len 32 --epochs 8 --batch_size 256
#
# Output:
# - prints metrics per epoch
# - saves best checkpoint to: <out_dir>/best.pt
#
# Checkpoint contains:
# - args
# - vocab
# - model_state

# Example usage:
# For donor model:
# python train_splice_single_task_kmer_bert.py \
#   --task donor \
#   --pos cnn_scoring/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
#   --neg cnn_scoring/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
#   --out_dir donor_model \
#   --k 3 --max_len 32 --epochs 8 --batch_size 256
# For acceptor model:
#python train_splice_single_task_kmer_bert.py \
#   --task acceptor \
#   --pos cnn_scoring/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pwm.err \
#   --neg cnn_scoring/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.neg.pwm.err \
#   --out_dir acceptor_model \
#   --k 3 --max_len 32 --epochs 8 --batch_size 256


import argparse
import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time


import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None


# --------------------------
# Repro utilities
# --------------------------
def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------
# Parsing your files
# --------------------------
# Supports three line formats:
#   DEBUG donor <SEQ> acceptor <SEQ> +
#   DEBUG pair <SEQ1> <SEQ2> +          (SEQ1=donor, SEQ2=acceptor)
#   DEBUG donor <SEQ> +
#   DEBUG acceptor <SEQ> -
#
# The final + or - is strand and ignored.
LINE_PAIR_NAMED_RE = re.compile(
    r"^DEBUG\s+donor\s+([A-Za-z]+)\s+acceptor\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_PAIR_RE = re.compile(
    r"^DEBUG\s+pair\s+([A-Za-z]+)\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)
LINE_SINGLE_RE = re.compile(
    r"^DEBUG\s+(donor|acceptor)\s+([A-Za-z]+)(?:\s+[+-])?\s*$"
)

TASK2ID = {"donor": 0, "acceptor": 1}


def read_examples_single_task(
    pos_paths: List[str],
    neg_paths: List[str],
    task: str,
) -> List[Tuple[str, int]]:
    """
    Returns list of (sequence, label) for ONE task only (donor or acceptor).

    Label convention:
      - pos_paths => label 1
      - neg_paths => label 0

    Strand (+/- at end of line) is ignored.
    """
    assert task in TASK2ID
    want_task = task  # "donor" or "acceptor"
    examples: List[Tuple[str, int]] = []

    def read_one_set(paths: List[str], label: int):
        nonlocal examples
        for path in paths:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("DEBUG"):
                        continue

                    # Paired format: "DEBUG donor SEQ acceptor SEQ"
                    m_pair = LINE_PAIR_NAMED_RE.match(line)
                    if m_pair:
                        donor_seq, acceptor_seq = m_pair.groups()
                        if want_task == "donor":
                            examples.append((donor_seq.upper(), label))
                        else:
                            examples.append((acceptor_seq.upper(), label))
                        continue

                    # Paired format: "DEBUG pair SEQ1 SEQ2" (SEQ1=donor, SEQ2=acceptor)
                    m_pair2 = LINE_PAIR_RE.match(line)
                    if m_pair2:
                        donor_seq, acceptor_seq = m_pair2.groups()
                        if want_task == "donor":
                            examples.append((donor_seq.upper(), label))
                        else:
                            examples.append((acceptor_seq.upper(), label))
                        continue

                    # Single format: one of donor/acceptor
                    m_single = LINE_SINGLE_RE.match(line)
                    if m_single:
                        tname, seq = m_single.groups()
                        if tname == want_task:
                            examples.append((seq.upper(), label))
                        continue

                    # Otherwise ignore unrecognized DEBUG formats

    read_one_set(pos_paths, label=1)
    read_one_set(neg_paths, label=0)
    return examples


# --------------------------
# k-mer tokenizer
# --------------------------
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def build_kmer_vocab(k: int = 3) -> Dict[str, int]:
    bases = ["A", "C", "G", "T"]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(vocab)

    def rec(prefix: str, depth: int):
        nonlocal idx
        if depth == k:
            vocab[prefix] = idx
            idx += 1
            return
        for b in bases:
            rec(prefix + b, depth + 1)

    rec("", 0)
    return vocab


def kmerize(seq: str, k: int) -> List[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def encode_kmers(kmers: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab["[UNK]"]
    return [vocab.get(km, unk) for km in kmers]


# --------------------------
# Dataset + Batch
# --------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, L) token ids
    attn_mask: torch.Tensor  # (B, L) 1 real, 0 pad
    labels: torch.Tensor     # (B,) 0 = neg example, 1 = pos example


class SpliceDataset(Dataset):
    def __init__(self, examples: List[Tuple[str, int]], vocab: Dict[str, int], k: int = 3, max_len: int = 32, task: str = "donor"):
        self.examples = examples
        self.vocab = vocab
        self.k = k
        self.max_len = max_len
        self.task = task

        self.pad_id = vocab["[PAD]"]
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq, y = self.examples[idx]
        # Clip raw sequence to max_len nucleotides before tokenization
        if len(seq) > self.max_len:
            if self.task == "donor":
                seq = seq[: self.max_len]     # keep left/5' flank
            else:  # acceptor
                seq = seq[-self.max_len :]    # keep right/3' flank
        kmers = kmerize(seq, self.k)
        ids = [self.cls_id] + encode_kmers(kmers, self.vocab) + [self.sep_id]

        # truncate
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.sep_id

        attn_mask = [1] * len(ids)

        # pad
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
            attn_mask += [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch) -> Batch:
    input_ids, attn_mask, labels = zip(*batch)
    return Batch(
        input_ids=torch.stack(input_ids, dim=0),
        attn_mask=torch.stack(attn_mask, dim=0),
        labels=torch.stack(labels, dim=0),
    )


# --------------------------
# Small BERT-like encoder + single head
# --------------------------
class SmallBertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Transformer expects True for PAD positions
        key_padding_mask = (attn_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x  # (B, L, D)


class SingleTaskSpliceModel(nn.Module):
    def __init__(self, encoder: SmallBertEncoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input_ids, attn_mask)  # (B, L, D)
        cls = x[:, 0, :]                        # (B, D) [CLS] token
        logits = self.head(cls).squeeze(-1)     # (B,)
        return logits


class FocalLoss(nn.Module):
    """Binary focal loss from logits, with optional pos_weight."""
    def __init__(self, gamma: float = 3.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# --------------------------
# Train / eval
# --------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attn_mask = batch.attn_mask.to(device)
        labels = batch.labels.to(device)

        logits = model(input_ids, attn_mask)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    logits = np.concatenate(all_logits) if all_logits else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])

    probs = sigmoid_np(logits) if logits.size else np.array([])

    out = {}
    if labels.size:

        # we will use pr_auc prefarably,
        #fallback to roc_auc if pr_auc not available,
        #else fallback to acc@0.5 (not great for imbalanced but better than nothing)

        out["acc@0.5"] = float(np.mean((probs >= 0.5) == (labels >= 0.5)))
        out["loss_proxy_bce"] = float(np.mean(-(labels * np.log(probs + 1e-9) + (1 - labels) * np.log(1 - probs + 1e-9))))
        if roc_auc_score is not None and len(np.unique(labels)) > 1:
            out["roc_auc"] = float(roc_auc_score(labels, probs))
        if average_precision_score is not None and len(np.unique(labels)) > 1:
            out["pr_auc"] = float(average_precision_score(labels, probs))
        out["n"] = int(labels.size)
        out["pos"] = int(labels.sum())
        out["neg"] = int(labels.size - labels.sum())
    return out


def stratified_split_by_label(examples: List[Tuple[str, int]], val_frac=0.1, seed=1337):
    """
    Stratify by label (0/1) so train/val both have positives and negatives.

    val_frac: fraction of examples to put in validation set (e.g. 0.1 for 90% train, 10% val)
    seed: random seed for shuffling
    """
    rng = random.Random(seed)
    buckets = {0: [], 1: []}
    for seq, y in examples:
        buckets[int(y)].append((seq, int(y)))

    train, val = [], []
    for y, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac)) if len(items) > 0 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main():
    #cmd line options
    ap = argparse.ArgumentParser()

    #Required: task + pos/neg data files
    ap.add_argument("--task", choices=["donor", "acceptor"], required=True, help="Which task to train")
    ap.add_argument("--pos", type=str, required=True, help="Path to POSITIVE file (.pwm.err)")
    ap.add_argument("--neg", type=str, required=True, help="Path to NEGATIVE file (.neg.pwm.err)")
    
    #Optional:output directory
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <task>_model)")
    
    #Optional: k-mer tokenization (default k=3)
    ap.add_argument("--k", type=int, default=3)
    
    #Optional: model hyperparameters
    ap.add_argument("--max_len", type=int, default=32) # The max allowed "context window"
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    
    #Optional: training hyperparameters
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    
    #Optional: validation split settings
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    
    #Optional: DataLoader settings
    ap.add_argument("--num_workers", type=int, default=2)
    
    #Optional: positive class weight for BCE loss (useful if imbalanced). If not set, will be computed from training split as neg/pos ratio.
    ap.add_argument("--pos_weight", type=float, default=None,
                    help="Optional: manual positive class weight for BCE (e.g. 6.0). "
                         "If not set, will be computed from training split as neg/pos.")
    
    #Optional: loss function
    ap.add_argument("--loss", choices=["bce", "focal"], default="focal",
                help="Loss function: bce (BCEWithLogitsLoss) or focal (Focal Loss)")
    ap.add_argument("--focal_gamma", type=float, default=3.0,
                help="Gamma for focal loss (only used when --loss focal)")

    #Optional: learning rate scheduler
    ap.add_argument("--scheduler", choices=["none", "cosine_warm"], default="none",
                help="LR scheduler: none (manual cosine, current default) or "
                     "cosine_warm (CosineAnnealingWarmRestarts with linear warmup)")
    ap.add_argument("--warmup_frac", type=float, default=0.05,
                help="Fraction of total steps for linear warmup (only with --scheduler cosine_warm)")

    #Optional: checkpoint filename
    ap.add_argument("--ckpt_name", type=str, default="best.pt",
                help="Checkpoint filename to save in out_dir (default: best.pt)")

    args = ap.parse_args()

    run_t0 = time.perf_counter()


    set_seed(args.seed)

    out_dir = args.out_dir or f"{args.task}_model"
    os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    pin = (device == "cuda")  # pin_memory is mainly useful for CUDA


    # Read examples for the selected task only
    examples = read_examples_single_task(
        pos_paths=[args.pos],
        neg_paths=[args.neg],
        task=args.task,
    )
    if len(examples) == 0:
        raise RuntimeError("No examples parsed. Check file formats / regex in script.")

    # Basic stats
    n_pos = sum(y for _, y in examples)
    n_neg = len(examples) - n_pos
    print(f"Task: {args.task}")
    print(f"Parsed examples: {len(examples)} | pos={n_pos} | neg={n_neg}")

    train_ex, val_ex = stratified_split_by_label(examples, val_frac=args.val_frac, seed=args.seed)
    train_pos = sum(y for _, y in train_ex)
    train_neg = len(train_ex) - train_pos
    val_pos = sum(y for _, y in val_ex)
    val_neg = len(val_ex) - val_pos
    print(f"Train: {len(train_ex)} | pos={train_pos} | neg={train_neg}")
    print(f"Val:   {len(val_ex)} | pos={val_pos} | neg={val_neg}")

    # Vocab + datasets
    vocab = build_kmer_vocab(args.k)
    print(f"Vocab size (k={args.k}): {len(vocab)} (incl specials)")

    train_ds = SpliceDataset(train_ex, vocab=vocab, k=args.k, max_len=args.max_len, task=args.task)
    val_ds = SpliceDataset(val_ex, vocab=vocab, k=args.k, max_len=args.max_len, task=args.task)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Model
    encoder = SmallBertEncoder(
        vocab_size=len(vocab),
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = SingleTaskSpliceModel(encoder, d_model=args.d_model).to(device)

    # Pos-weight handling (optional but often useful if imbalanced)
    if args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
    else:
        # default: neg/pos computed from TRAIN split
        pos_weight = (train_neg / max(1, train_pos)) if train_pos > 0 else 1.0
        pos_weight = np.sqrt(pos_weight)  # use sqrt of ratio as a softer weighting (tune as needed)
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"Using FocalLoss(gamma={args.focal_gamma}")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        print(f"Using BCEWithLogitsLoss(pos_weight={pos_weight:.4f})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # ---- Initial (pre-training) validation metrics ----
    init_val_metrics = evaluate(model, val_loader, device=device)
    print("\nVal metrics before any training:", init_val_metrics)

    if "pr_auc" in init_val_metrics:
        init_score = init_val_metrics["pr_auc"]
        init_score_name = "pr_auc"
    elif "roc_auc" in init_val_metrics:
        init_score = init_val_metrics["roc_auc"]
        init_score_name = "roc_auc"
    else:
        init_score = init_val_metrics.get("acc@0.5", 0.0)
        init_score_name = "acc@0.5"
    print(f"Model score before training ({init_score_name}): {init_score:.4f}\n")

    # LR schedule setup
    total_steps = args.epochs * max(1, len(train_loader))

    if args.scheduler == "cosine_warm":
        warmup_steps = max(1, int(total_steps * args.warmup_frac))
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # linear warmup
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.5 * (1.0 - 0.05) * (1 + math.cos(math.pi * progress))  # cosine decay to 5% of peak
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"Using cosine_warm scheduler: {warmup_steps} warmup steps, {total_steps} total steps")
    else:
        # Original manual cosine (no warmup, decays to 10% of peak)
        scheduler = None
        def lr_at(step):
            min_lr = args.lr * 0.1
            return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))

    best_val_score = -1e9
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attn_mask = batch.attn_mask.to(device)
            labels = batch.labels.to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if scheduler is not None:
                scheduler.step()
            else:
                lr = lr_at(global_step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device=device)

        if "pr_auc" in val_metrics:
            score = val_metrics["pr_auc"]
            score_name = "pr_auc"
        elif "roc_auc" in val_metrics:
            score = val_metrics["roc_auc"]
            score_name = "roc_auc"
        else:
            score = val_metrics.get("acc@0.5", 0.0)
            score_name = "acc@0.5"


        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        if val_metrics:
            print(f"  Val metrics: {val_metrics}")
        print(f"  Model score ({score_name}): {score:.4f}")

        if score > best_val_score:
            best_val_score = score
            ckpt = {
                "args": vars(args),
                "vocab": vocab,
                "model_state": model.state_dict(),
            }
            out_path = os.path.join(out_dir, args.ckpt_name)
            torch.save(ckpt, out_path)
            print(f" ----- Saved best checkpoint to: {out_path} ------")

    print("\nDone.")
    print(f"Best {args.task} model score: {best_val_score:.4f}")
    print(f"Checkpoint: {os.path.join(out_dir, args.ckpt_name)}")
    run_t1 = time.perf_counter()
    print(f"Total run time: {run_t1 - run_t0:.2f} seconds")




if __name__ == "__main__":
    main()
