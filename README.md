# Information Maximizing Self Augmented Training (IMSAT)
This is a reproducing code for IMSAT [1]. IMSAT is a method for discrete representation learning using deep neural networks. It can be applied to clustering and hash learning to achieve the state-of-the-art results. This is the work performed while Weihua Hu was interning at Preferred Networks.

## Requirements 
You must have the following already installed on your system.
- Python 2.7
- Chainer 1.21.0, sklearn, munkres

## Quick start
For reproducing the experiments on MNIST datasets in [1], run the following codes.
- Clustering with MNIST: ``` python imsat_cluster.py ```
- Hash learning with MNIST: ``` python imsat_hash.py ```

`calculate_distance.py` is be used to calculate the perturbation range for Virtual Adversarial Training [2]. For MNIST dataset, we have already calculated the range.

## Reference ##
[1] Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto and Masashi Sugiyama. Learning Discrete Representations via Information Maximizing Self Augmented Training.

[2] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, and Shin Ishii. Distributional smoothing with virtual adversarial training. In ICLR, 2016.
