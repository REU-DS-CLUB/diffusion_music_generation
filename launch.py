import sys
import torch
from torch.distributed import launch

if __name__ == "__main__":
    num_gpus = 4
    gpus_per_process = 2
    num_processes = num_gpus // gpus_per_process

    sys.argv.extend(["--nproc_per_node", str(gpus_per_process)])

    launch.main(
        main_python_path="ddpm.py",
        node_rank=0,
        num_nodes=1,
        nproc_per_node=gpus_per_process,
        func=None,
        args=None,
        rdzv_backend=None,
        rdzv_endpoint=None,
        rdzv_configs=None,
        start_method="spawn",
    )
