## The code has been adapted from https://github.com/kevinzakka/recurrent-visual-attention


The *Recurrent Attention Model* (RAM) is a neural network that processes inputs sequentially, attending to different locations within the image one at a time, and incrementally combining information from these fixations to build up a dynamic internal representation of the image.


## Model Description


In this paper, the attention problem is modeled as the sequential decision process of a goal-directed agent interacting with a visual environment. The agent is built around a recurrent neural network: at each time step, it processes the sensor data, integrates information over time, and chooses how to act and how to deploy its sensor at the next time step.


# The data can be downloaded from https://drive.google.com/drive/folders/1D_u1vKUL87Ubhivv8GjmVr2TFRDqw0W9?usp=sharing


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
