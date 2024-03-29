# (MI)llion-(M)ax(M)argin (MI-MM)
Code and Data of the MIllion-MaxMargin model used as baseline for the [EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge](https://competitions.codalab.org/competitions/26138#learn_the_details).

The MI-MM model is a modified version of the model used in [End-to-End Learning of Visual Representations from Uncurated Instructional Videos](https://www.di.ens.fr/willow/research/mil-nce/). The main changes are three:
1) Using tho Gated Embedding Unit instead of a simple Linear layer to project the features pace into the embedding space (read [Learning a Text-Video Embedding from
Incomplete and Heterogeneous Data](https://arxiv.org/pdf/1804.02516.pdf) to know more about the Gated Embedding Unit).
2) Increasing the size of the vocabulary due to the lack of wsome relevant words.
3) Using the Multi-instance MaxMargin loss instead of the MIL-NCE loss (read [Fine-Grained Action Retrieval Through Multiple Parts-of-Speech Embeddings](https://arxiv.org/pdf/1908.03477.pdf) to know more about the Multi-instance MaxMargin loss).

<p align="center">
  <img width="378" height="393" src="https://github.com/adrianofragomeni/MI-MM/blob/main/img/model.png?raw=true">
</p>

## Data
The data directory contains 5 folders:
* `dataframes`: It contains the train and validation csv files of EPIC-Kitchens.
* `features`: It contains the training and validation features of EPIC-Kitchens extracted by the S3D model trained on HowTo100M whose implementation can be found [here](https://github.com/antoine77340/S3D_HowTo100M). The order of the videos is the same as in the csv files you can find in the `dataframes` folder.
* `models`: It contains the weights of the MI-MM model.
* `relevancy`: It contains the training relevancy matrix needed to train the model.
* `resources`: It contains the weights of the S3D model trained on HowTo100M and the word embeddings.

You can download the data directory from [here](https://www.dropbox.com/sh/5gl70rk7qznw4cs/AAAjJHVQyMB3BHLbVeGOdVo2a?dl=0)

## How to use it

### Requirements
You can install a conda enviroment using `conda env create -f environment.yml`.

### Training
You can train the model using the default values defined in `src/utils/utils.py` by running the the following command: `python training.py`. You can run `python training.py --help` to see the full list of arguments you can pass to the script.

During training you can check the loss value of each epoch in `data/models/logs`, where you can find `train.txt` and `val.txt`. Moreover, you can inspect the training and testing loss curves by running `tensorboard --logdir=data/models/logs/runs`.

After training the model, you can find 2 different weigths in `data/models`:
1) `model_best.pth.tar`: This file contains the weights of the model of the epoch with the lowest validation loss value.
2) `checkpoint.pth.tar`: This file contains the weights of the model of the last epoch.

### Testing
You can evaluate the model and create the submission file by running `python testing.py`. This will evaluate the model using the default value `model_best.pth.tar`, but you can select `checkpoint.pth.tar` by running `python testing.py --best-model=checkpoint.pth.tar`.

After testing, you can find the submission file in `output/test.pkl`.

Other details on the submission can be found [here](https://github.com/epic-kitchens/C5-Multi-Instance-Retrieval).
