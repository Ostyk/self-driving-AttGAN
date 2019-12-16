# self-driving-AttGAN
- Tensorflow implementation of ![Attribute-Controlled Traffic Data Augmentation Using Conditional Generative Models](docs/attn-gan.pdf)

## Pre-requisites
Revamped for the following usage:
* Python 3.6
* Tensorflow-gpu 1.14

## Custom Training
```
$ python code/train.py
    --batch_size 128
    --num_classes 2
    --generation_rate 2
    --lr_g 0.0002
    --lr_d 0.005
    --model_name None
    --truncated False
    --graph_path None
    --rand_seed 42
```

## sample run for 100 epochs
![](docs/latest.gif)




