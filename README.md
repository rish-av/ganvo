# GANVO
Generative Adversarial Network for Visual Odometry
*Pytorch Implementation of the paper -- [GANVO](https://arxiv.org/pdf/1809.05786.pdf)*

> **Disclaimer:** This is not an official release. This implementation is based on the ICRA 2019 paper of the same title by Sen Wang, Ronald Clark, Hongkai Wen, and Niki Trigoni. I am trying to reproduce the results, while incorporating my own interpretations of the approach, wherever needed. I plan to finish this by end of december 2021 -- I use collab for open-source projects, so it will take some time.


The work uses GANs for unsupervised visual odometry which is later used for depth estimation.


## Training
- Prepare the config file -- this specifies the various hyperparameters, weights directory, summary, the root directory of the dataset, etc.
- Install all the dependencies with `pip install -r requirements.txt`
- Run `python train.py --config ./config.yaml`
- For pretrained weights visit this [link](#)

## Testing
- Testing Pose -- run `python eval_pose.py --config ./eval_pose.py` for quantitative pose evaluation.

## Inference
- For depth inference on a single image, run `python inference.py --img_path ./test_image.png --generator_weights ./weights/generator.pth --h 128 --w 416`
## Qualitative Results
**Dec Last Week**
## Quantitative Results
**Dec Last Week**
