# SS-BERT

## Training

Add the training data to `train_data/...`, edit `scripts/runall.sh` to match the parameters you want to train the model with (and also choose the model you want to train), and simply pass it to bash:

```bash
bash scripts/runall.sh
```

Currently, `runall.sh` also contains reference to the testing code, so it's better you also generate the test data before you run `runall.sh` (details below).

## Testing

First add the FASTA and GTF files inside `raw_files/...`, then run `scripts/make_test_data.sh` with the correct files and sequence lengths. The test data should now be generated in the indicated directory.

