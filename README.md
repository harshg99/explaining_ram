## Explaining Recurrent Attention Models
(The code has been adapted from https://github.com/kevinzakka/recurrent-visual-attention)


Recurrent Attention Models (RAM) actively selects and observes a sequence of patches in an image to make a
prediction. Unlike in the deep convolution network, in hard attention it is explainable which regions of the image contributed
to the prediction. To infer the glimpses and explain the model qualitatively, we build a Variational Autoencoder (VAE) on the
final hidden state of the recurrent units and visualize the reconstruction of the images after each glimpse is processed. We
also prove quantitatively the model encodes some latent space statistics of the entire image through a sequence of patches by
evaluating the expected information gain(EIG) over the classification output after each glimpse. These are demonstrated on the
MNIST and cluttered MNIST dataset. We also attempted to study the improvement in the above statistics through reward
shaping the inherent reinforcement learning algorithm that dictates where to see next. We report that the new reward structure
performs better than the original one used in the paper in terms of information gain over the MNIST dataset however, no
improvement was reported in terms of expected information gain.

## Model Description


In this paper, the attention problem is modeled as the sequential decision process of a goal-directed agent interacting with a visual environment. The agent is built around a recurrent neural network: at each time step, it processes the sensor data, integrates information over time, and chooses how to act and how to deploy its sensor at the next time step.


The data can be downloaded from https://drive.google.com/drive/folders/1D_u1vKUL87Ubhivv8GjmVr2TFRDqw0W9?usp=sharing

## Network Description

![image](https://user-images.githubusercontent.com/28558013/209129175-7c7bd290-80f1-4a33-ae95-f9e167e08733.png)

![image](https://user-images.githubusercontent.com/28558013/209129483-0e0790ff-e1b2-4d07-8c76-1eb440f701e5.png)


## Usage


The easiest way to start training your RAM variant is to edit the parameters in `arguments.py` and run the following command:
Please create the following folders prior to running the code within the same directory as the code:
1)ckpt
2)data
3)logs
4)models
5)plots
6)report
7)tests

```
python main.py
```
To resume training, run:
```
python main.py --resume=True
```
Finally, to test a checkpoint of your model that has achieved the best validation accuracy, run the following command:
```
python main.py --is_train=False
```

## References


- [Torch Blog Post on RAM](http://torch.ch/blog/2015/09/21/rmva.html)
- https://proceedings.neurips.cc/paper/2014/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf
