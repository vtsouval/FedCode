import os
import torch
import torchvision
import numpy as np
import contextlib

def get_cifar10_dataset(split_fn=None, id=0, num_shards=1, return_eval_ds=False,
		batch_size=32, seed=42):

	gen = torch.Generator().manual_seed(seed)
	# Prepare train data
	train_prep = torchvision.transforms.Compose([
			torchvision.transforms.RandomCrop(32, padding=4),
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
	])
	with contextlib.redirect_stdout(None):
		ds_train = torchvision.datasets.CIFAR10(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'),
						train=True, download=True, transform=train_prep)
	# Prepare test data
	test_prep = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
	])
	with contextlib.redirect_stdout(None):
		ds_test = torchvision.datasets.CIFAR10(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'),
						train=False, download=True, transform=test_prep)
	# Get dataset info
	samples_idxs = np.array(ds_train.targets)
	num_classes = len(np.unique(samples_idxs))

	if return_eval_ds:
		ds_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
		return ds_test, num_classes, len(ds_test)

	# Split per client
	samples_idxs = split_fn(idxs=samples_idxs, num_shards=num_shards,
						num_samples=len(ds_train.targets), num_classes=num_classes, seed=seed)[id]
	ds_train = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train,samples_idxs),
					batch_size=batch_size, shuffle=True, generator=gen)
	return ds_train , num_classes, len(samples_idxs)

def get_cifar100_dataset(split_fn=None, id=0, num_shards=1, return_eval_ds=False,
		batch_size=32, seed=42):

	gen = torch.Generator().manual_seed(seed)
	# Prepare train data
	train_prep = torchvision.transforms.Compose([
		torchvision.transforms.RandomCrop(32, padding=4),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),])
	with contextlib.redirect_stdout(None):
		ds_train = torchvision.datasets.CIFAR100(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'),
						train=True, download=True, transform=train_prep)
	# Prepare test data
	test_prep = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),])
	with contextlib.redirect_stdout(None):
		ds_test = torchvision.datasets.CIFAR100(os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'),
						train=False, download=True, transform=test_prep)
	# Get dataset info
	samples_idxs = np.array(ds_train.targets)
	num_classes = len(np.unique(samples_idxs))

	if return_eval_ds:
		ds_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
		return ds_test, num_classes, len(ds_test)

	# Split per client
	samples_idxs = split_fn(idxs=samples_idxs, num_shards=num_shards,
						num_samples=len(ds_train.targets), num_classes=num_classes, seed=seed)[id]
	ds_train = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train,samples_idxs),
					batch_size=batch_size, shuffle=True, generator=gen)
	return ds_train , num_classes, len(samples_idxs)