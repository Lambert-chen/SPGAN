# SP-GAN: Self-growing and Pruning GenerativeAdversarial Networks
## Note
Each file has a brief introduction function and operation mode.
## Platform
python3.6  TensorFlow-GPU >= 1.14.0   numpy->=1.13.3  opencv-python>=3.4.0.12  We recommend Anaconda3.
Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
## Abstract
This paper presents a new Self-growing and Pruning Generative Adversarial Network (SP-GAN) for realistic image generation. In contrast to traditional GAN models, our SPGAN is able to dynamically adjust the size and architecture of a network in the training stage, by using the proposed self-growing and pruning methods. To be more specific, we first train two seed networks as the generator and discriminator, each only
8 contains a small number of convolution kernels. Such small scale networks are much easier to train than large capacity networks.Second, in the self-growing step, we replicate the convolution kernels of each seed network to augment the scale of the network,followed by retraining of the augmented/expanded network. More importantly, to prevent the excessive growth of each seed network in the self-growing stage, we propose a pruning strategy that reduces the redundancy of an augmented network, yielding the optimal scale of the network. Last, we design an iterative
17 loss function that is treated as a variable loss computational process to train the proposed SP-GAN model. By design, the hyper-parameters of the loss function can constantly adapt to different training levels.
## SP-GAN framework
![image](https://github.com/Lambert-chen/SPGAN/blob/master/image/frame.png)
## Feature map visualization
![image](https://github.com/Lambert-chen/SPGAN/blob/master/image/Feature_map.png)

If you have any question about this code, feel free to reach me(yao_chen@aliyun.com)
