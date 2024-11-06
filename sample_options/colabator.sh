#!/usr/bin/env bash


HF_HUB_OFFLINE=True CUDA_VISIBLE_DEVICES=4,5,6,7  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=4396 corun_colabator/train.py -opt sample_options/colabator_sample.yml --launcher pytorch