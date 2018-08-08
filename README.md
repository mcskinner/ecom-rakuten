# ecom-rakuten
Winning solution for the Rakuten Data Challenge, as part of SIGIR eCom '18.

The details of the model choices and evolution can be found in the [system description paper](https://sigir-ecom.github.io/ecom18DCPapers/ecom18DC_paper_9.pdf) for that workshop.

## Usage

1. **Prepare the data.** Run a train/test split, build the vocabularies, and save the int-encoded training and validation sets for later.
1. **Train the BPV models.** Run the model training script one or more times, which saves the model weights somewhere. The hyperparameters default to those in the final RDC solution, but can be adjusted via command line flags. The exception is the flag for training on a reversed data set, which the winning solution used for half of the networks.
1. **Ensemble the models and infer.** Run the inference script by pointing it to the saved models and a test file for which to generate predictions. By default this will also tune the F1 score for each category, but that can be disabled.

In code, from the repository root and assuming that `rdc-catalog-train.tsv` and `rdc-catalog-test.tsv` are in a `data/` subdirectory:
1. `./prep.sh` performs a train-validation split, tokenizes the data, builds the vocabularies, and saves the processed data back to new files in `data/`.
1. `./train.sh model-name` trains a forward model with the default hyperparameters and training schedule, and then saves the model weights to `data/models/model-name.h5`. To perform bi-directional ensembling, the `--reverse` flag can be used to train on a backwards version of the input sequences.
1. Model scoring, ensembling, test set inference are not yet implemented. Soon.
