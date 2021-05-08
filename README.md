# (MI)llion-(M)ax(M)argin (MI-MM)
Code and Data of the MIllion-MaxMargin model used as baseline for the EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge.

## Data
The data directory contains 5 folders:
* `dataframes`: It contains the train and validation csv files of EPIC-Kitchens.
* `features`: It contains the training and validation features of EPIC-Kitchens extracted by the S3D model trained on HowTo100M whose implementation can be found [here](https://github.com/antoine77340/S3D_HowTo100M). The order of the videos is the same as in the csv files you can find in the `dataframes` folder.
* `models`: It contains the weights of the MI-MM model.
* `relevancy`: It contains the training relevancy matrix needed to train the model.
* `resources`: It contains the weights of the S3D model trained on HowTo100M and the word embeddings.

You can download the data directory from [here](https://www.dropbox.com/sh/lp1zu27e9dbemfi/AADankJuhiOurXqYk3bXTGLRa?dl=0)

## How to use it

### Training
