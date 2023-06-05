#!/bin/bash
#SBATCH --job-name=train-nn-gpu
#SBATCH -t 00:30:00                  # set time limit
#SBATCH -p grete:interactive         # -p grete:shared for training, -p grete:interactive for debugging 


#SBATCH -G V100:1                    # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
##SBATCH --mem-per-gpu=5G             # setting the right constraints for the splitted gpu partitions

#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task

## mail settings for job info
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=j.rieling@stud.uni-goettingen.de  # mailaddress
#SBATCH --output=./slurm_files/slurm-%x-%j.out        # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err         # where to write slurm error

module load anaconda3
module load cuda
source activate dnlp



# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script(s):
python -u classifier.py --option pretrain --use_gpu --lr 1e-3 --batch_size 64 --hidden_dropout 0.3
python -u classifier.py --option finetune --use_gpu --lr 1e-5 --batch_size 64 --hidden_dropout 0.3

