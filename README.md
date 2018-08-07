# ecom-rakuten
Winning solution for the Rakuten Data Challenge, as part of SIGIR eCom '18.

The details of the model choices and evolution can be found in the [system description paper](https://sigir-ecom.github.io/ecom18DCPapers/ecom18DC_paper_9.pdf) for that workshop.

## Usage

1. **Prepare the data.** Run a train/test split, build the vocabularies, and save the int-encoded training and validation sets for later.
1. **Train the BPV models.** Run the model training script one or more times, which saves the model weights somewhere. The hyperparameters default to those in the final RDC solution, but can be adjusted via command line flags. The exception is the flag for training on a reversed data set, which the winning solution used for half of the networks.
1. **Ensemble the models and infer.** Run the inference script by pointing it to the saved models and a test file for which to generate predictions. By default this will also tune the F1 score for each category, but that can be disabled.
