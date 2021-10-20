# GANVO
Generative Adversarial Network for Visual Odometry
*Unofficial Pytorch Implementation of the paper -- [GANVO](https://arxiv.org/pdf/1809.05786.pdf)*

The work uses GANs for unsupervised visual odometry which is later used for depth estimation.

## Training
- Prepare the config file -- this specifies the various hyperparameters, weights directory, summary, the root directory of teh dat, etc.
- Install all the dependencies with `pip install -r requirements.txt`
- Run `python train.py --config ./config.yaml`
- For pretrained weights visit this [link](#)

# Testing
- To generate visuals from the validation set, setup a test config file (eg. test_config.yaml)
- Run `python test.py --config ./test_config.yaml`

## Qualitative Results

## Quantitative Results