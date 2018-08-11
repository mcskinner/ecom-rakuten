# ecom-rakuten
Winning solution for the Rakuten Data Challenge, as part of SIGIR eCom '18.

The details of the model choices and evolution can be found in the [system description paper](https://sigir-ecom.github.io/ecom18DCPapers/ecom18DC_paper_9.pdf) for that workshop.

## Usage

**Data preparation.**

Set up the expected data directories, from the repository root:

`mkdir -p data/models`

Move the challenge files into the `data/` subdirectory:

`mv path/to/rdc-catalog-train.tsv data/`
`mv path/to/rdc-catalog-test.tsv data/`

Run a train/test split, build the vocabularies, and save the int-encoded training and validation sets for later:

`./prep.sh`

**BPV model training.**

Train and save a forward model with the hyperparameters from the winning RDC solution (the model goes in `data/models/model-name.h5`):

`./train.sh model-name`

Train a reverse model, intended for use in building a bi-directional ensemble with a forward network:

`./train-sh reverse-model --reverse`

**Inference, prediction, and scoring.**

Run an inference on the validation set, generate predictions, and then output precision, recall, and F1:

`./infer.sh model-name`

or

`./infer.sh --forward=model-name` 

Score a reverse model:

`./infer.sh --reverse=reverse-model`

Similarly for a bi-directional ensemble:

`./infer.sh --forward=model-name --reverse=reverse-model` 

Or for a larger ensemble, e.g. with 4 each forward and reverse:

`./infer.sh --forward=fwd1,fwd2,fwd3,fwd4 --reverse=rev1,rev2,rev3,rev4` 

To run test set inference and output prediction files:

`./infer.sh model-name --is-test`
