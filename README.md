# SuperMarioBros.github.io

This package creates an AI that is able to play the original Super Mario Bros. It uses a Double Deep Q Network Reinforcement Learning algorithm to do this. It uses a combination of a Convolutional Neural Network and a Deep Q Learning Neural Network to select the best actions (Q-values) for the Reinforcement Learning algorithm.

## Installation

**To clone this repository**

```bash
git clone https://github.com/BlakeMuchmore01/SuperMarioBros.github.io.git
```

**Next, create a virtual environment**

The command below is to create a conda environment. I'm using Python 3.10.12.

```bash
conda create --name smbrl python=3.10.12
```

To activate the environment.

```bash
conda activate smbrl
```

**Installing PyTorch v2.1.1**

The command below is to install pytorch
```bash
pip install torch
```

**Finally, install the rest of the requirements**

```bash
pip install -r requirements.txt
```