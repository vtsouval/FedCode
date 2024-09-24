import os
import torch
import torchaudio
import numpy as np
import collections
import sklearn
import contextlib

__CLASSES__ = collections.defaultdict(lambda: 0, {j:i for i,j in enumerate('unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', '))})

class AudioTransform(torch.nn.Module):

	def __init__(self, length=15360, mels=64, n_fft=400,hop_length=160, sample_rate=16000):
		super(AudioTransform, self).__init__()
		self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=n_fft,hop_length=hop_length,n_mels=mels,f_min=125.,f_max=7500.,pad=0,power=2.0, normalized=False,)
		#self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
		self.length = length

	@staticmethod
	def add_log_offset(x, offset=1e-3):
		return x + offset

	@staticmethod
	def center_pad_or_trim_audio(waveform, length):
		if waveform.shape[1] < length:
			padding = length - waveform.shape[1]
			left_pad = padding // 2
			right_pad = padding - left_pad
			waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
		elif waveform.shape[1] > length:
			start = (waveform.shape[1] - length) // 2
			waveform = waveform[:, start : start + length]
		return waveform

	def forward(self, x):
		x = __class__.center_pad_or_trim_audio(x, self.length)
		x = self.spectrogram(x)
		x = __class__.add_log_offset(x, 1e-3)
		return x

class SpeechCommands(torchaudio.datasets.SPEECHCOMMANDS):

	def __init__(self, transform=None, *args, **kwargs):
		super(SpeechCommands, self).__init__(*args, **kwargs)
		self.transform = transform #AudioTransform(length=16000)
		self.num_classes = len(__CLASSES__)
		self.targets = [__CLASSES__[self.get_metadata(i)[2]] for i in range(len(self))]

	def __getitem__(self, n):
		metadata = self.get_metadata(n)
		waveform = torchaudio.datasets.utils._load_waveform(self._archive, metadata[0], metadata[1])
		if self.transform is not None: waveform = self.transform(waveform)
		return waveform, __CLASSES__[metadata[2]]

	def make_weights_for_balanced_classes(self):
		return sklearn.utils.class_weight.compute_class_weight('balanced',
					classes=np.unique(np.array(self.targets)), y=np.array(self.targets))

def get_spcm_dataset(split_fn=None, id=0, num_shards=1, return_eval_ds=False, batch_size=32, seed=42):

	gen = torch.Generator().manual_seed(seed)

	# Prepare train data
	data_prep = AudioTransform(length=16000)
	with contextlib.redirect_stdout(None):
		ds_train = SpeechCommands(root=os.environ['TORCH_DATA_DIR'], url='speech_commands_v0.02', \
					folder_in_archive='spcm', subset='training', download=True, transform=data_prep,)
	# Prepare test data
	with contextlib.redirect_stdout(None):
		ds_test = SpeechCommands(root=os.environ['TORCH_DATA_DIR'], url='speech_commands_v0.02', \
					folder_in_archive='spcm',subset='testing', download=True, transform=data_prep,)

	# Get dataset info
	samples_idxs = np.array(ds_train.targets) #np.array([__CLASSES__[train_ds.get_metadata(i)[2]] for i in range(len(train_ds))])
	num_classes = len(np.unique(samples_idxs))
	#print(ds_train.make_weights_for_balanced_classes())

	if return_eval_ds:
		ds_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
		return ds_test, num_classes, len(ds_test)

	# Split per client
	samples_idxs = split_fn(idxs=samples_idxs, num_shards=num_shards,
						num_samples=len(ds_train.targets), num_classes=num_classes, seed=seed)[id]
	ds_train = torch.utils.data.DataLoader(torch.utils.data.Subset(ds_train,samples_idxs),
					batch_size=batch_size, shuffle=True, generator=gen,)
	return ds_train , num_classes, len(samples_idxs)

'''
if __name__ == "__main__":
	get_spcm_dataset(None)
'''