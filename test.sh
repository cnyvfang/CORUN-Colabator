

CUDA_VISIBLE_DEVICES=4  python3  corun_colabator/simple_test.py \
  --opt options/test_corun.yml \
  --input_dir /home/nfs/fcy/Datasets/Dehaze/RTTS/JPEGImages  \
  --result_dir ./results/ \
  --weights ./pretrained_weights/15k.pth \
  --dataset RTTS_15K