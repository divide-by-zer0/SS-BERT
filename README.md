# SS-BERT: Transformer-based Splice Site Scoring for Genome Annotation

<!--**Contributors:** Chirag Adwani, Hibiki Kato, Aleksey Zimin, James A. Yorke, Yoshitaka Saiki-->

---

## Overview

Accurate genome annotation depends critically on identifying and scoring
splice sites — the donor and acceptor boundaries within transcripts where
introns are removed during RNA splicing. Current state-of-the-art annotation
tools such as EviANN rely on Hidden Markov Models (HMMs) for transcript
scoring. HMMs are fundamentally limited by the Markov property: they model
only local sequence context and cannot capture long-range dependencies in
nucleotide sequences.

SS-BERT replaces HMM-based scoring with a BERT-style Transformer encoder
trained from scratch on genomic sequences. By attending across the full
transcript, SS-BERT captures long-range dependencies that HMMs structurally
cannot, achieving substantially improved splice site scoring over HMM baselines.

This project is a step toward a broader goal: a full transcript annotation
model capable of identifying multiple genomic features with higher sensitivity
and precision than existing tools.

---

## The Problem

Pre-mRNA splicing requires precise identification of donor (5') and acceptor
(3') splice sites within transcripts. Errors in splice site prediction
directly degrade the quality of genome annotation, affecting downstream
biological analysis.

Existing tools use HMMs to score candidate transcripts — assigning
probabilities to putative splice site positions based on local sequence
patterns. While effective at capturing short-range motifs, HMMs cannot model
dependencies between distant positions in a transcript, a known limitation
for this task.

---

## Approach

SS-BERT frames splice site scoring as a sequence classification problem:

- **Input:** Nucleotide sequences surrounding candidate splice sites,
  tokenized at the character level (A, T, G, C, N)
- **Model:** BERT-style encoder (multi-head self-attention + feedforward
  layers), trained from scratch — no pre-training on external genomic
  databases
- **Output:** Scores for donor and acceptor splice site candidates within
  a transcript
- **Training:** Supervised on labeled genomic sequence data; trained on
  GPU clusters using PyTorch

The use of full self-attention allows the model to condition predictions
on the entire input sequence, directly addressing the locality limitation
of HMMs.

---

## Repository Structure
```
SS-BERT/
├── models/          # Model architecture definitions
├── scripts/         # Training and evaluation shell scripts
├── evaluate/        # Evaluation code
├── train_data/      # Add training data here (not included)
├── test_data/       # Add test data here (not included)
├── raw_files/       # FASTA and GTF input files (not included)
└── README.md
```

---

## Usage

### Training

Add training data to `train_data/`, configure parameters in
`scripts/runall.sh`, then run:
```bash
bash scripts/runall.sh
```

### Testing

Add FASTA and GTF files to `raw_files/`, generate test data:
```bash
bash scripts/make_test_data.sh <fasta_file> <gtf_file> <sequence_length>
```

Test data will be generated in the specified output directory.
`runall.sh` also invokes the testing pipeline, so generate test
data before running it.

---

## Notes

Training data and model weights are not included in this repository.
Results and full evaluation will be reported in an upcoming publication.

---

## Dependencies

- Python 3.x
- PyTorch
- Standard bioinformatics utilities for FASTA/GTF processing

---

## Citation

A paper describing this work is in preparation. In the meantime, if you
use or reference this repository, please cite:
```
Adwani et al., SS-BERT: Transformer-based Splice Site Scoring for
Genome Annotation (in preparation)
```

---

## Contact

Chirag Adwani — cadwani@umd.edu


<!-- 
# SS-BERT

## Training

Add the training data to `train_data/...`, edit `scripts/runall.sh` to match the parameters you want to train the model with (and also choose the model you want to train), and simply pass it to bash:

```bash
bash scripts/runall.sh
```

Currently, `runall.sh` also contains reference to the testing code, so it's better you also generate the test data before you run `runall.sh` (details below).

## Testing

First add the FASTA and GTF files inside `raw_files/...`, then run `scripts/make_test_data.sh` with the correct files and sequence lengths. The test data should now be generated in the indicated directory. 

-->


