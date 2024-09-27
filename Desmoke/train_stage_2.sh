#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=4,5,6,7  python3 -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=4397 DASUM/train.py -opt Desmoke/stage_2_restormer.yml   --launcher pytorch