# BERT and Multitask BERT for Sentiment Analysis, Semantic Similarity and Paraphrase detection

This is our group repository for the final project for the Deep Learning for Natural Language Processing course at the University of Göttingen SS23 ([Full instructions here](https://1drv.ms/b/s!AkgwFZyClZ_qk718ObYhi8tF4cjSSQ?e=3gECnf))

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

Anaconda or Miniconda is required. For setup of the project run:

```setup
$~: git clone https://gitlab.gwdg.de/yangshan.xiang/do-nlp-with-linguistic-mavericks.git
$~: ./minibert-default-final-project/setup.sh 
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the sentiment classifier from part 1 of the project, run:

```train
$~: conda activate dnlp
$~: python classifier.py --option pretrain --epochs 10 --lr 1e-3 --batch_size 64 --hidden_dropout_prob 0.3
$~: python classifier.py --option finetune --epochs 10 --lr 1e-5 --batch_size 64 --hidden_dropout_prob 0.3
```

To run any form of training on the GWDG HPC cluster: 

```
# Connect to frontend node
you@local   ~$: ssh -i path/to/key <user>@glogin9.hlrn.de
<user>@node ~$: cd /scratch/usr/<user>/

# If not already done, set up the conda environment and clone this repository
<user>@node ~$: cd do-nlp-with-linguistic-mavericks/minbert-default-final-project/

# specify the desired node and node configuration in submit_train.sh
#   Consider changing:
#       - Time Limit/ desired node and configuration
#       - desired training command to be executed (!! Dont forget to use '--use-gpu' flag !!)
#       - E-Mail for receiving updates
<user>@node ~$: vim submit_train.sh # incorporate desired changes but do not commit them in here 

# submit job
<user>@node ~$: sbatch submit_train.sh
```

To view the training process, you can see the output in the `slurm_files`folder in `<job_id>.out and .err`files.
You will receive details about your job and the name of the processing node on the email you wrote in `submit_train.sh`
Additionally, you can connect to the node running the training from the frontend node using and see live statistics of your job:
```
<user>@node         ~$: ssh <name of processing node>
<user>@process_node ~$: module load nvitop
<user>@process_node ~$: nvitop
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Methodology

required section as per project description section 9.3. 

## Experiments

required section as per project description section 9.3.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## Individual Contributions 

required section as per project description section 9.3. 

### Part 1

#### Yangshan, De Xiong Michael, Leander, Jonas 
* we all implemented the reqired code independently for learning purposes
* for our main branch, we merged Michaels version of the optimizer and Leanders self-attention mechanism

### Part 2

#### Yangshan
* Contribution 1
* ...

#### De Xiong Michael
* Contribution 1
* ...

#### Leander
* Contribution 1
* ...

#### Jonas
* Contribution 1
* ...

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 