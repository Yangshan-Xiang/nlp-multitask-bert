# BERT & Convolutional Neural Networks for Multitask Training
This repository is for the default final project for the Deep Learning for Natural Language Processing course at the University of Göttingen. You can find the handout [here](https://1drv.ms/b/s!AkgwFZyClZ_qk718ObYhi8tF4cjSSQ?e=3gECnf)

In this repository, you can review the implementation of [BERT](https://arxiv.org/pdf/1810.04805.pdf) & [Convolutional Neural Networks](https://arxiv.org/pdf/1408.5882.pdf) tuned for multitask training.
- BERT model generates initial embeddings for words and sentences which is used in multitask training.
- TextCNN processes embeddings and output predictions for sentiment analysis, semantic similarity and paraphrase detection.
<p align="center">
<img src="etc/TextCNN.png" height=350>
</p>

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Run
- Training, evaluation and testing process are all integrated in the [multitask_classifier.py](multitask_classifier.py)
- To run this model on GWDG remote cluster, enter the compute nodes, copy this repository to remote cluster, then run this command in the terminal:
```
sbatch train.sh
```
- A detailed description of the code structure is in [STRUCTURE.md](STRUCTURE.md).

## Pre-trained Models

The pretrained model is `finetune-100-1e-05-multitask.pt`, you can find it in this repository.

- The model is trained on `ids-sst-train.csv`, `quora-train.csv` and `sts-train.csv`.
- With hyperparameters `lr` = 1e-5, `batch_size` = 64, `hidden_dropout_prob` = 0.3, `epochs` = 100.
- Actual trained `epochs` = 31, due to early stopping and time limit.
- The parameters are taken from the 27th epoch, as they produced the highest average accuracy.

## Results

The model is trained on NVIDIA A100 Tensor Core GPU for 10 hours, achieved the following performance for the three different tasks:


| Task name              | Training Accuracy | Development Accuracy |
|------------------------|-------------------|----------------------|
| Sentiment Analysis     | 98.6%             | 49.7%                |
| Semantic Similarity    | 98.9%             | 71.3%                |
| Paraphrase Detection   | 99.9%             | 86.0%                |

The accuracy over 31 epochs is displayed in the graph below:
<p align="center">
<img src="etc/accuracy_over_epochs.png" height=350>
</p>

You can find the model predictions under the [predictions](predictions) folder and the running information under the [slurm_files](slurm_files) folder.
## Experiments
Tried different loss functions for semantic similarity:
- cosine similarity: Noticed the evaluation of semantic similarity task is by computing Pearson correlation, so using cosine similarity as loss function can achieve a better correlation on development set, but the predictions are unreasonable and out of the range between 0 and 5.
- mean square error: In order to solve the unreasonable predictions, picked MSE as loss function, predictions are within the range, but the Pearson correlation decreased.

## Methodology
The core of this model is the `TextCNN` which is inspired by computer vision.

CNN has achieved state-of-art performance in computer vision tasks which take images as input data, in order to take advantages of CNN, we can consider our word embeddings as single-channel grayscale images.

We know that the output of `BertModel` has two parts, `last_hidden_state` and `pooler_output`, 
the idea is to use `TextCNN` to process `last_hidden_state` to extract the local relations between words, 
due to the local receptive field, using CNN allows the model to focus more on local neighboring words, 
and use simple neural network to process `pooler_output` to extract the global features.

The input of the `TextCNN` is a concatenation of `last_hidden_state` and the pretrained word embeddings, as the pretrained word embedding certainly contains a lot of useful information, and it requires no gradient during training.
Then feed the input into three convolution layers with different kernel size, as kernel size determines the size of local receptive field, this can allow the model to extract features from different scales.
Next in order to process different size of input, Spatial Pyramid Pooling is added to this model.
Finally use fully connected layer to generate the desired output for loss function.

## Contribution
Due to the unique architecture (using CNN) of this model, the work in this repository is mainly completed by myself ([Yangshan Xiang](https://gitlab.gwdg.de/yangshan.xiang)), especially the implementation of `TextCNN`.

There are few ideas which I adopted from my group mates:
- From [Jonas](https://gitlab.gwdg.de/j.rieling): The architecture for processing `pooler_output`.
- From [Leander](https://gitlab.gwdg.de/leander.booms01): Give `multitask_classifier.py` the flexibility to train any combination of different tasks.

## Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!) 

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).


## References
- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
- [Sentiment Analysis: Using Convolutional Neural Networks](https://d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-cnn.html#the-textcnn-model)
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)