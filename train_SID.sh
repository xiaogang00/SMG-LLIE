#!/bin/sh




python -m torch.distributed.launch --nproc_per_node=4 --master_port=4507 scripts/train_SID.py \
--dataset_type ours_encode \
--exp_dir experiment/SID \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 1.0 \
--progressive_start 20000 \
--id_lambda 0.5 \
--val_interval 5000 \
--max_steps 800000 \
--stylegan_size 256 \
--stylegan_weights pretrained_models/e4e_cars_encode.pt \
--workers 1 \
--batch_size 1 \
--test_batch_size 1 \
--test_workers 1 \
--save_training_data \
--save_interval 5000 \
--optim_name adam \
--w_discriminator_lr 0.0002 \
--learning_rate 0.0002
