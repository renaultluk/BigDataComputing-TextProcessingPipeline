# Text Processing Pipeline

As document volumes grow across diverse domains like news articles, social media, and scientific literature, traditional single-machine implementations for text analysis become impractical due to memory and processing limitations. This project addresses the challenge of efficiently extracting insights from such large document collections by developing and evaluating a scalable, distributed text processing pipeline using Apache Spark. Our core focus is on topic modeling, for which we implement a distributed version of Latent Dirichlet Allocation (LDA). This report details the algorithmic foundations, our Spark-based implementation strategy, the experimental evaluation of its performance and topic quality, and discusses key findings.

This repository is designed to be run on the [HKUST SuperPOD cluster](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod).

Prerequisites
---
Please ensure that you have access to the SuperPOD cluster, either using the eduroam network  or connecting through [Pulse Secure](https://itso.hkust.edu.hk/services/cyber-security/vpn).

Setup
---
1. Once you have connected to the SuperPOD cluster through SSH and cloned this repository within the login node, run the following commands to  retrieve the PySpark running environment in this directory. The resulting image is stored in the generated `spark-py_latest.sif` file.
```bash
$ module load apptainer
$ apptainer pull docker://apache/spark-py
```

2. Schedule the job onto the cluster:
```bash
$ sbatch {job name}.sbatch
```
There are 3 jobs defined in this repository, and the file names are aligned accordingly:

Job Name | Description
--- | ---
msbd5003 | Preprocessing to TF-IDF flow on the [Text Document Classification Dataset](https://www.kaggle.com/datasets/sunilthite/text-document-classification-dataset/data)
msbd5003_medium | LDA on the [20 Newsgroups dataset](https://www.kaggle.com/datasets/crawford/20-newsgroups/data) with all scalability tests and hyperparameter tuning
msbd5003_medium_1b | LDA on the [20 Newsgroups dataset](https://www.kaggle.com/datasets/crawford/20-newsgroups/data) with core model logic and evaluation only

3. Any outputs or logs from the runtime can be found in the `slurm.out` or `lda_job_{job ID}.out` file, and any errors can be found in `lda_job_{job ID}.err`.

Configuring Runtime
---
For configuring PySpark parameters, you can edit the `run.sh` file, which contains the `spark-submit` command which kickstarts the cluster and jobs.

For configuring SuperPOD resources, you can edit the `sbatch` files according to the target job, which contains parameters defined in [#SBATCH tags](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job).