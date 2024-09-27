#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0,1,2,3  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=4396 DASUM/train.py -opt Sample_Options/colabator_with_transmission_sample.yml --launcher pytorch