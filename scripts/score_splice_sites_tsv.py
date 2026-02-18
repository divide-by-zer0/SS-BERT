#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------
# Device selection (CUDA > MPS > CPU)
# --------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------
# Tokenizer (same as training)
# --------------------------
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]


def kmerize(seq: str, k: int) -> List[str]:
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(0, len(seq) - k + 1)]


def encode_kmers(kmers: List[str], vocab: Dict[str, int]) -> List[int]:
    unk = vocab.get("[UNK]", 3)
    return [vocab.get(km, unk) for km in kmers]


# --------------------------
# Dataset for scoring
# --------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attn_mask: torch.Tensor


class SeqDataset(Dataset):
    def __init__(self, seqs: List[str], vocab: Dict[str, int], k: int, max_len: int):
        self.seqs = seqs
        self.vocab = vocab
        self.k = k
        self.max_len = max_len

        self.pad_id = vocab["[PAD]"]
        self.cls_id = vocab["[CLS]"]
        self.sep_id = vocab["[SEP]"]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].strip().upper()
        kmers = kmerize(seq, self.k)
        ids = [self.cls_id] + encode_kmers(kmers, self.vocab) + [self.sep_id]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            ids[-1] = self.sep_id

        attn_mask = [1] * len(ids)

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
            attn_mask = attn_mask + [0] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn_mask, dtype=torch.long),
        )


def collate_fn(batch) -> Batch:
    input_ids, attn_mask = zip(*batch)
    return Batch(
        input_ids=torch.stack(input_ids, dim=0),
        attn_mask=torch.stack(attn_mask, dim=0),
    )


# --------------------------
# Model (must match training)
# --------------------------
class SmallBertEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
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
        key_padding_mask = (attn_mask == 0)  # True where PAD
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class SingleTaskSpliceModel(nn.Module):
    def __init__(self, encoder: SmallBertEncoder, d_model: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input_ids, attn_mask)  # (B, L, D)
        cls = x[:, 0, :]                        # (B, D)
        logits = self.head(cls).squeeze(-1)     # (B,)
        return logits


def load_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    vocab = ckpt["vocab"]

    encoder = SmallBertEncoder(
        vocab_size=len(vocab),
        max_len=int(args["max_len"]),
        d_model=int(args["d_model"]),
        n_heads=int(args["n_heads"]),
        n_layers=int(args["n_layers"]),
        dropout=float(args["dropout"]),
    )
    model = SingleTaskSpliceModel(encoder, d_model=int(args["d_model"]))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    k = int(args["k"])
    max_len = int(args["max_len"])
    return model, vocab, k, max_len


@torch.no_grad()
def predict_probs(model: nn.Module, vocab: Dict[str, int], k: int, max_len: int,
                  seqs: List[str], device: str, batch_size: int, num_workers: int):
    ds = SeqDataset(seqs, vocab=vocab, k=k, max_len=max_len)
    pin = (device == "cuda")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        collate_fn=collate_fn,
        drop_last=False,
    )

    probs = []
    for batch in dl:
        input_ids = batch.input_ids.to(device)
        attn_mask = batch.attn_mask.to(device)
        logits = model(input_ids, attn_mask)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    if probs:
        return np.concatenate(probs).astype(np.float32)
    return np.array([], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True, help="Input test TSV produced from GTF/FASTA")
    ap.add_argument("--donor_ckpt", required=True, help="Path to donor_model/best.pt")
    ap.add_argument("--acceptor_ckpt", required=True, help="Path to acceptor_model/best.pt")
    ap.add_argument("--out_tsv", required=True, help="Output TSV with probabilities")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=0)  # macOS safest
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = ap.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    print(f"Scoring device: {device}")

    # Load models
    donor_model, donor_vocab, donor_k, donor_max_len = load_checkpoint(args.donor_ckpt, device=device)
    acc_model, acc_vocab, acc_k, acc_max_len = load_checkpoint(args.acceptor_ckpt, device=device)

    # Safety: vocab should match (it will, given same k + specials)
    if donor_k != acc_k:
        raise RuntimeError(f"k mismatch: donor k={donor_k}, acceptor k={acc_k}")
    if donor_vocab != acc_vocab:
        # If dict equality fails due to ordering differences, it still might be semantically same,
        # but since we saved vocab, we should just require exact match.
        raise RuntimeError("Vocab mismatch between donor and acceptor checkpoints.")

    # Read input TSV rows
    rows = []
    donor_seqs = []
    donor_idx = []
    acc_seqs = []
    acc_idx = []

    with open(args.in_tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        required = {"transcript_id", "gene_id", "site_type", "intron_index", "chrom", "strand", "boundary_pos", "seq"}
        if not required.issubset(set(r.fieldnames or [])):
            raise RuntimeError(f"Input TSV missing columns. Need at least: {sorted(required)}")

        for i, row in enumerate(r):
            site_type = row["site_type"].strip().lower()
            seq = row["seq"].strip().upper()

            rows.append(row)
            if site_type == "donor":
                donor_idx.append(i)
                donor_seqs.append(seq)
            elif site_type == "acceptor":
                acc_idx.append(i)
                acc_seqs.append(seq)
            else:
                raise RuntimeError(f"Unknown site_type '{row['site_type']}' on line {i+2}")

    # Predict
    probs_out = [None] * len(rows)

    if donor_seqs:
        d_probs = predict_probs(
            donor_model, donor_vocab, donor_k, donor_max_len,
            donor_seqs, device=device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        for idx, p in zip(donor_idx, d_probs):
            probs_out[idx] = float(p)

    if acc_seqs:
        a_probs = predict_probs(
            acc_model, acc_vocab, acc_k, acc_max_len,
            acc_seqs, device=device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        for idx, p in zip(acc_idx, a_probs):
            probs_out[idx] = float(p)

    # Write output TSV (same columns + prob)
    out_fields = ["transcript_id", "gene_id", "site_type", "intron_index", "chrom", "strand", "boundary_pos", "seq", "prob"]
    with open(args.out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
        w.writeheader()
        for row, p in zip(rows, probs_out):
            out_row = {k: row.get(k, "") for k in out_fields}
            out_row["prob"] = f"{p:.6f}" if p is not None else ""
            w.writerow(out_row)

    print(f"Done. Wrote scores to: {args.out_tsv}")


if __name__ == "__main__":
    main()
