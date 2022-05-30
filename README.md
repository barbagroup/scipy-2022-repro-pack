Reproducibility package for SciPy 2022 proceeding submission
============================================================

Cases under folder `petibm` were run with the Singularity image created by `petibm-0.5.4rc2-hpcx207-cuda102.singularity` under `singularity` folder.
Cases under folder `modulus` were run with the Singularity image created by `modulus-22.03.singularity` under `singularity` folder.
However, Modulus is not an open-source software.
You need to go to NVIDIA's developer zone, download Modulus 22.03, and follow the instruction to build a Docker image in the local registry.
Then, you can use `modulus-22.03.singularity` to create the Singularity image.

**PetIBM**

Basically just go into each case folder and run

```shell
$ CUDA_VISIBLE_DEVICES=<list of target gpus> \
    mpiexec \
        -n <number of CPUs> \
            singularity exec --nv <petibm singularity image> petibm-navierstokes
```

The results shown in the paper were obtained using 1 single K40c GPU and 6 CPU cores of i7-5930K.


**Modulus**

Each case has `job.sh` for Slurm scheduler.
Modify the script for your target cluster, and then do `sbatch job.sh` to submit each job one by one.

The resource used (the partition shown in `job.sh`) was a node of NVIDIA DGX-A100-640G.

**Post-processing**

First, generate processed data by executing the following three python scripts: `modulus_cylinder_2d_re200.py`, `modulus_tgv_2d_re100.py`, `petibm_tgv_2d_re100.py`.
Then generate figures using other scripts.
Generated figures will be in a folder called `figures`.
