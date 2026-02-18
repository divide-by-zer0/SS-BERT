# SS-BERT

## Training

Add the training data to `train_data/...`, edit `scripts/runall.sh` to match the parameters you want to train the model with (and also choose the model you want to train), and simply pass it to bash:

```bash
bash scripts/runall.sh
```

Currently, `runall.sh` also contains reference to the testing code, so it's better you also generate the test data before you run `runall.sh` (details below).

## Testing

First add the FASTA and GTF files inside `raw_files/...`, then run `scripts/make_test_data.sh` with the correct files and sequence lengths. The test data should now be generated in the indicated directory.

## Note

Please make sure to check `scripts/score_splice_sites_tsv.py` if it matches the hyperparameters of your trained model (`--d_model`, `--n_heads`, `--n_layers`). It does not automatically take those numbers as of right now, this is something that needs to be fixed.
