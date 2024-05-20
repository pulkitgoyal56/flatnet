# *flatnet*

> Results of experiments with entropy regularization, using a toy CNN classifier for pixelated MNIST images.  

Entropy regularization in given in the context of a multi-label image classification problem between labels,

```math
    L(\bm{i}, \bm{l}) = L_{\iota}(\bm{i}, \bm{l}) \underbrace{+ \kappa \sum_{\mathbf{i}_k \in \bm{i}}\sum_{\mathbf{l}_j \in \bm{l}} p(\mathbf{l}_j; \mathbf{i}_k, \bm{l}) \log p(\mathbf{l}_j; \mathbf{i}_k, \bm{l})}_{\text{Entropy Regularization}},
```

where $L_{\iota}(\bm{i}, \bm{l})$ is the original loss of the model given a batch of input images $\bm{i}$ and labels $\bm{l}$, $\kappa$ is the regularization strength, and $p(\mathbf{l}_j; \mathbf{i}_k, \bm{l})$ is the probability of the $j$-th label given the $k$-th image.

The context of this is using such models as a source of rewards for Reinforcement Learning.
The original application of this was to fine-tune CLIP models so that they have less noise and their semantics entropy reward trajectories are smoother.
The expectation is that this would lead to denser rewards, decreased semantic bias in random images (misclassifications) / improved specificity, and possibly reduced class preference in CLIP's outputs.

---

This repository presents the results of using this entropy regularization in tandem with an augmented training dataset (using random images that should ideally be classified with equal probability among the labels) to train an MNIST classifier.

> [Click here to see the model summary and comparisons](./info/Flatnet%20Models%20Summary.md).  
> The results can be visualized [here](https://www.sharecanvas.io/p/flatnet-comparison).  
> For architecture details of *flatnet*, please see [the code](./flatnet.py).
