"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach_LOL_synthetic import Coach


def synchronize():
	if not dist.is_available():
		return
	if not dist.is_initialized():
		return
	world_size = dist.get_world_size()
	if world_size == 1:
		return
	dist.barrier()


def init_dist(opts, backend='nccl'):
	torch.cuda.set_device(opts.local_rank)
	torch.distributed.init_process_group(backend="nccl", init_method="env://")
	synchronize()


def main():
	opts = TrainOptions().parse()
	previous_train_ckpt = None

	init_dist(opts)
	world_size = torch.distributed.get_world_size()
	rank = torch.distributed.get_rank()
	coach = Coach(opts, previous_train_ckpt)
	coach.test_edge_generation()


def load_train_checkpoint(opts):
	train_ckpt_path = opts.resume_training_from_ckpt
	device_id = torch.cuda.current_device()
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location=lambda storage, loc: storage.cuda(device_id))
	new_opts_dict = vars(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = Namespace(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt


def setup_progressive_steps(opts):
	log_size = int(math.log(opts.stylegan_size, 2))
	num_style_layers = 2 * log_size - 8
	num_deltas = num_style_layers - 1
	if opts.progressive_start is not None:  # If progressive delta training
		opts.progressive_steps = [0]
		next_progressive_step = opts.progressive_start
		for i in range(num_deltas):
			opts.progressive_steps.append(next_progressive_step)
			next_progressive_step += opts.progressive_step_every

	assert opts.progressive_steps is None or is_valid_progressive_steps(opts, num_style_layers), \
		"Invalid progressive training input"


def is_valid_progressive_steps(opts, num_style_layers):
	return len(opts.progressive_steps) == num_style_layers and opts.progressive_steps[0] == 0


def create_initial_experiment_dir(opts):
	'''
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	'''
	if not(os.path.exists(opts.exp_dir)):
		os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
	for k, v in new_opts.items():
		if k not in ckpt_opts:
			ckpt_opts[k] = v
	if new_opts['update_param_list']:
		for param in new_opts['update_param_list']:
			ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
	main()
