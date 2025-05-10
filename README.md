# Text Processing Pipeline

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
$ sbatch -o slurm.out msbd5003.sbatch
```

3. Any outputs or logs from the runtime can be found in the `slurm.out` file.

Configuring Runtime
---
For configuring PySpark parameters, you can edit the `run.sh` file, which contains the `spark-submit` command which kickstarts the cluster and jobs.

For configuring SuperPOD resources, you can edit the `msbd5003.sbatch` file, which contains parameters defined in [#SBATCH tags](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/submit-first-job).