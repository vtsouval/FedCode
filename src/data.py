from datasets.cifar import get_cifar10_dataset as cifar10_dataloader
from datasets.cifar import get_cifar100_dataset as cifar100_dataloader
from datasets.spcm import get_spcm_dataset as spcm_dataloader


def get_dataset_fn(name='cifar10'):
	if name=='cifar10':
		return cifar10_dataloader
	elif name=='cifar100':
		return cifar100_dataloader
	elif name=='spcm':
		return spcm_dataloader
	else:
		raise ValueError(f"Unknown dataset name: {name}")
