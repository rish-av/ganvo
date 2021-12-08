# GANVO
Generative Adversarial Network for Visual Odometry
*Unofficial Pytorch Implementation of the paper -- [GANVO](https://arxiv.org/pdf/1809.05786.pdf)*

The work uses GANs for unsupervised visual odometry which is later used for depth estimation.

I am currently training the network -- The weights alongside other quantitative and qualitative stuff will be made available by Dec 24 2021. (I am poor--can only use collab :-P and also have a full-time job), feel free to use this code though!

# Training
- Prepare the config file -- this specifies the various hyperparameters, weights directory, summary, the root directory of the dataset, etc.
- Install all the dependencies with `pip install -r requirements.txt`
- Run `python train.py --config ./config.yaml`
- For pretrained weights visit this [link](#)

# Testing
- To generate visuals from the validation set, setup a test config file (eg. test_config.yaml)
- Run `python test.py --config ./test_config.yaml`

## Qualitative Results

## Quantitative Results
