#!/usr/bin/env python3

'''
To run:

python min_intron_sum.py --in_tsv test_sites.scored-4.8M.tsv --out_tsv transcript_min_scores-4.8M.tsv
'''

import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--dedup", choices=["first", "max", "mean"], default="first",
                    help="If duplicates exist for same (transcript_id,intron_index,site_type), how to combine.")
    ap.add_argument("--combine", choices=["sum", "product"], default="product",
                    help="How to combine donor and acceptor scores into a single intron score.")
    
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t")

    required = {"transcript_id", "intron_index", "site_type", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # ensure numeric
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")

    # optional: handle duplicates per key
    key = ["transcript_id", "intron_index", "site_type"]
    if args.dedup == "mean":
        df = df.groupby(key, as_index=False)["prob"].mean()
    elif args.dedup == "max":
        df = df.groupby(key, as_index=False)["prob"].max()
    else:
        df = df.drop_duplicates(key, keep="first")

    # pivot donor/acceptor into columns per intron
    intr = df.pivot_table(
        index=["transcript_id", "intron_index"],
        columns="site_type",
        values="prob",
        aggfunc="first",
    ).reset_index()

    # make sure we got both columns
    if "donor" not in intr.columns or "acceptor" not in intr.columns:
        # Some introns might be missing a donor or acceptor; we can drop them.
        # (You can change this behavior if desired.)
        pass

    intr = intr.rename(columns={"donor": "Score_donor", "acceptor": "Score_acceptor"})

    if args.combine == "product":
        intr["intron_score"] = intr["Score_donor"] * intr["Score_acceptor"]
        score_col = "donor_times_acceptor"
    else:
        intr["intron_score"] = intr["Score_donor"] + intr["Score_acceptor"]
        score_col = "donor_plus_acceptor"

    # drop introns missing either score (so combined score is valid)
    intr = intr.dropna(subset=["intron_score"]).copy()

    # transcript-level min over introns
    idx = intr.groupby("transcript_id")["intron_score"].idxmin()
    out = intr.loc[idx, ["transcript_id", "intron_index", "Score_donor", "Score_acceptor", "intron_score"]].copy()
    out = out.rename(columns={
        "intron_index": "min_intron_index",
        "intron_score": f"min_{score_col}",
    }).sort_values("transcript_id")

    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out)} transcripts to {args.out_tsv}")
    
if __name__ == "__main__":
    main()
