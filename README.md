Reproducibility package for SciPy 2022 proceeding submission
============================================================

**Title**: "Experience report of physics-informed neural networks in fluid simulations: pitfalls and frustration"
**Preprint on arXiv**: https://arxiv.org/abs/2205.14249

Cases under folder `petibm` were run with the Singularity image created by `petibm-0.5.4rc2-hpcx207-cuda102.singularity` under `singularity` folder.
Cases under folder `modulus` were run with the Singularity image created by `modulus-22.03.singularity` under `singularity` folder.
However, Modulus is not an open-source software.
You need to go to NVIDIA's developer zone, download Modulus 22.03, and follow the instruction to build a Docker image in the local registry.
Then, you can use `modulus-22.03.singularity` to create the Singularity image.
(Update: at the time when we ran these cases, we needed to create the Docker image manually.
However, Modulus has provided a Docker image on the NGC platform, so there's no need to manually build it now.)

**PetIBM**

Basically just go into each case folder and run

```shell
$ CUDA_VISIBLE_DEVICES=<list of target gpus> \
    mpiexec \
        -n <number of CPUs> \
            singularity exec --nv <petibm singularity image> petibm-navierstokes
```

The results shown in the paper were obtained using 1 single K40c GPU and 6 CPU cores of i7-5930K.
For cylinder flow, replace `petibm-navierstokes` with `petibm-decoupledibpm`.


**Modulus**

Each case has `job.sh` for Slurm scheduler.
Modify the script for your target cluster, and then do `sbatch job.sh` to submit each job one by one.

The original resource used (the partition shown in `job.sh`) was a node of NVIDIA DGX-A100-640G, but it should work on other GPUs.
Just note the memory usage of some cases may be non-trivial, and probably only A100 (both 40GB and 80GB variants) can host all runs.
V100 32GB may be able to handle most cases except some extreme ones.

**Post-processing**

First, generate processed data by executing the following three python scripts: `modulus_cylinder_2d_re200.py`, `modulus_tgv_2d_re100.py`, `petibm_tgv_2d_re100.py`.
Then generate figures using other scripts.
Generated figures will be in a folder called `figures`.
