#!/usr/bin/env bash

HF_HUB_OFFLINE=True CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port=4396 corun_colabator/train.py -opt option_templates/stage1_restormer_sample.yml --launcher pytorch