#!/bin/bash
#SBATCH --job-name=lda-test          # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --gpus=2                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=mscbdt2024

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
module load apptainer

pip install pyspark nltk

apptainer run spark-py_latest.sif bash /home/wlluk/msbd5003/run.sh
