#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=6  python3 -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port=4396 corun_colabator/train.py -opt options/train_corun_with_depth.yml --launcher pytorch