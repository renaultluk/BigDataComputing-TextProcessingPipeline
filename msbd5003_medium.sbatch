#!/bin/bash
#SBATCH --job-name=lda-test          # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # tasks per node
#SBATCH --cpus-per-task=8            # CPUs per task (max allowed in cpu partition)
#SBATCH --mem=32G                    # memory per node (reduced to fit quota)
#SBATCH --time=08:00:00              # total run time limit (HH:MM:SS)
#SBATCH --partition=cpu              # use cpu partition for CPU-based job
#SBATCH --account=mscbdt2024        # your account
#SBATCH --output=lda_job_%j.out      # output file
#SBATCH --error=lda_job_%j.err       # error file

module purge                         # clear environment modules
module load Anaconda3/2023.09-0      # load Anaconda
module load apptainer                # load Apptainer

# Log environment for debugging
echo "Running on node: $(hostname -i)"
echo "Loaded modules:"
module list

# Copy dataset to local disk for faster I/O
echo "Copying dataset to /tmp"
cp /home/shgchu/df_Mid_Size.csv /tmp/df_Mid_Size.csv

# Set Spark configurations for performance
export SPARK_WORKER_CORES=8
export SPARK_WORKER_MEMORY=28g
export SPARK_DEFAULT_PARALLELISM=16

# Install required Python packages inside the Apptainer container (only if not already installed)
if [ ! -d "/home/shgchu/.local/lib/python3.10/site-packages/pyspark" ]; then
    apptainer exec spark-py_latest.sif python3 -m pip install --user pyspark==3.4.0 nltk numpy matplotlib
fi

# Pre-download NLTK data (only if not already downloaded)
if [ ! -d "/home/shgchu/nltk_data" ]; then
    apptainer exec spark-py_latest.sif python3 -c "import nltk; nltk.data.path.append('/home/shgchu/nltk_data'); nltk.download('stopwords'); nltk.download('wordnet')"
fi

# Run the Spark job in local mode with optimized settings
echo "Running Spark job in local mode"
apptainer run spark-py_latest.sif bash -c "/opt/spark/bin/spark-submit \
    --master local[*] \
    --driver-memory 6g \
    --executor-memory 6g \
    --conf spark.default.parallelism=16 \
    --conf spark.sql.shuffle.partitions=16 \
    MSBD5003_Project_medium.py"