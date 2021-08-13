# Knowledge Tracing Collection with PyTorch

This repository is a collection of the following knowledge tracing algorithms:
- **Deep Knowledge Tracing (DKT)**
- **Dynamic Key-Value Memory Networks for Knowledge Tracing (DKVMN)**
- **A Self-Attentive model for Knowledge Tracing (SAKT)**
- **A Stacked variant of SAKT (SSAKT)**

More algorithms will be added on this repository soon.

In this repository, [ASSISTment2009](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data) "skill-builder" dataset are used. You need to download the dataset on the following path:

```
.datasets/ASSIST2009/
```

Also, you can use the [ASSISTment2015](https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data) "skill-builder" dataset. Similarly you need to download them on the following path:

```
.datasets/ASSIST2015/
```

Other datasets, [Algebra 2005-2006](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp) and [Statics 2011](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507) dataset can be used to train your knowledge tracing model. The pathes to download each dataset are as follows:

```
.datasets/Algebra2005
```

```
.datasets/Statics2011
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
    - `learning_rate`: The learning of the optimizer for the training process. Default: 0.001
    - `optimizer`: The optimizer to use in the training process. The possible optimizers are ["sgd", "adam"]. Default: "adam"
    - `seq_len`: The sequence length for the dataset to use in the training process. Default: 100
2. Execute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python train.py --model_name=dkvmn
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```

## Training Results

### ASSISTment2009 Result
![](assets/img/2021-08-13-13-37-58.png)

### ASSISTment2015 Result
![](assets/img/2021-08-13-13-39-02.png)

### Algebra 2005-2006 Result
![](assets/img/2021-08-13-13-42-17.png)

### Statics 2011 Result
![](assets/img/2021-08-13-13-42-43.png)

The fact that `Adam Optimizer` has better performance on the training of DKT and DKVMN can be checked easily by running this repository.

SAKT looks like suffering an over-fitting. It seems that other tools to decrease the over-fitting will help the performance of SAKT. In fact, the results show that the dropout methods can relieve the over-fitting of the performance of SAKT.

## Recent Works
- The preprocess module for the two datasets: Algebra 2005-2006, Statics 2011 was updated.
- The results for Algebra 2005-2006 dataset and Statics 2011 dataset are updated.

## References
- DKT: [Deep Knowledge Tracing](https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf)
- DKVMN: [Dynamic Key-Value Memory Networks for Knowledge Tracing](https://arxiv.org/pdf/1611.08108.pdf)
- SAKT: [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/pdf/1907.06837.pdf)
- For the implementation of SAKT: [PyTorch Transforme Encoder Layer](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)
- The first introduction of SSAKT: [Towards an Appropriate Query, Key, and Value
Computation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf)
