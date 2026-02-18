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
    intr["sum_donor_acceptor"] = intr["Score_donor"] + intr["Score_acceptor"]

    # drop introns missing either score (so sum is valid)
    intr = intr.dropna(subset=["sum_donor_acceptor"]).copy()

    # transcript-level min over introns
    idx = intr.groupby("transcript_id")["sum_donor_acceptor"].idxmin()
    out = intr.loc[idx, ["transcript_id", "intron_index", "Score_donor", "Score_acceptor", "sum_donor_acceptor"]].copy()
    out = out.rename(columns={
        "intron_index": "min_intron_index",
        "sum_donor_acceptor": "min_donor_plus_acceptor"
    }).sort_values("transcript_id")

    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Wrote {len(out)} transcripts to {args.out_tsv}")

if __name__ == "__main__":
    main()
