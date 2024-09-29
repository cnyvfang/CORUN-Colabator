#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=4,5,6,7  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=4396 corun_colabator/train.py -opt options/train_corun_with_depth.yml --launcher pytorch