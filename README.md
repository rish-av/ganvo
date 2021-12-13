# GANVO
Generative Adversarial Network for Visual Odometry
*Unofficial Pytorch Implementation of the paper -- [GANVO](https://arxiv.org/pdf/1809.05786.pdf)*

The work uses GANs for unsupervised visual odometry which is later used for depth estimation.

I am currently training the network -- The weights alongside other quantitative and qualitative stuff will be made available by Dec Last Week. (I am poor--can only use collab :-P), feel free to use this code though!

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
