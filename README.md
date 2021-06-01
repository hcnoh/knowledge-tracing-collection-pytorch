# Knowledge Tracing Collection with PyTorch

This repository is a collection of the following knowledge tracing algorithms:
- **Deep Knowledge Tracing (DKT)**
- **Dynamic Key-Value Memory Networks for Knowledge Tracing (DKVMN)**
- **A Self-Attentive model for Knowledge Tracing (SAKT)**
- **A Stacked variant of SAKT (SSAKT)**

More algorithms will be added on this repository soon.

In this repository, [ASSISTments2009](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data) "skill-builder" dataset are used. You need to download the dataset on the following path:

```
.datasets/assistments2009/
```

Also, you can use the [ASSISTments2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) "skill-builder" dataset. Similarly you need to download them on the following path:

```
.datasets/assistments2015/
```

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```

3. Install PyTorch. The version of PyTorch should be greater or equal than 1.7.0. This repository provides the CUDA usage.

## Training and Running
1. Modify `config.json` as your machine setting. The following explanations are for understanding `train_config` of `config.json`:
    - `batch_size`: The batch size of the training process. Default: 64
    - `num_epochs`: The number of epochs of the training process. Default: 100
    - `train_ratio`: The ratio of the training dataset to split the whole dataset. Default: 0.9
    - `learning_rate`: The learning of the optimizer for the training process. Default: 1e-3
    - `optimizer`: The optimizer to use in the training process. The possible optimizers are ["sgd", "adam"]. Default: "adam"
    - `seq_len`: The sequence length for the dataset to use in the training process. Default: 200
2. Execute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python train.py --model_name=dkvmn
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```

## Training Results

### ASSISTments2009 Result
![](/assets/img/README/README_2021-05-21-09-16-23.png)

### ASSISTments2015 Result
![](/assets/img/README/README_2021-06-01-10-24-23.png)

The fact that `Adam Optimizer` has better performance on the training of DKT and DKVMN can be checked easily by running this repository.

SAKT looks like suffering an over-fitting. It seems that other tools to decrease the over-fitting will help the performance of SAKT. In fact, the results show that the dropout methods can relieve the over-fitting of the performance of SAKT.

## Recent Works
- The result for ASSISTments2015 dataset was uploaded.
- A new model(SSAKT) was added on this repository.

## References
- DKT: [Deep Knowledge Tracing](https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf)
- DKVMN: [Dynamic Key-Value Memory Networks for Knowledge Tracing](https://arxiv.org/pdf/1611.08108.pdf)
- SAKT: [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837.pdf)
- For the implementation of SAKT: [PyTorch Transforme Encoder Layer](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)
- The first introduction of SSAKT: [Towards an Appropriate Query, Key, and Value
Computation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf)
