import torch

CKPT_URL = "https://github.com/w-hc/torch_audioset/releases/download/v0.1/yamnet.pth"

class Conv2d_tf(torch.nn.Conv2d):

	def __init__(self, *args, **kwargs):
		padding = kwargs.pop("padding", "SAME")
		super().__init__(*args, **kwargs)
		self.padding = padding
		assert self.padding == "SAME"
		self.num_kernel_dims = 2
		self.forward_func = lambda input, padding: torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, padding=padding, dilation=self.dilation, groups=self.groups,)

	def tf_SAME_padding(self, input, dim):
		print(input.size(0), input.size(1), input.size(2))
		input_size = input.size(dim)
		filter_size = self.kernel_size[dim]
		dilate = self.dilation
		dilate = dilate if isinstance(dilate, int) else dilate[dim]
		stride = self.stride
		stride = stride if isinstance(stride, int) else stride[dim]
		effective_kernel_size = (filter_size - 1) * dilate + 1
		out_size = (input_size + stride - 1) // stride
		total_padding = max(0, (out_size - 1) * stride + effective_kernel_size - input_size)
		total_odd = int(total_padding % 2 != 0)
		return total_odd, total_padding

	def forward(self, input):
		if self.padding == "VALID": return self.forward_func(input, padding=0)
		odd_1, padding_1 = self.tf_SAME_padding(input, dim=0)
		odd_2, padding_2 = self.tf_SAME_padding(input, dim=1)
		if odd_1 or odd_2: input = torch.nn.functional.pad(input, [0, odd_2, 0, odd_1])
		return self.forward_func(input, padding=[ padding_1 // 2, padding_2 // 2 ])

class CONV_BN_RELU(torch.nn.Module):

	def __init__(self, conv):
		super().__init__()
		self.conv = conv
		self.bn = torch.nn.BatchNorm2d(conv.out_channels, eps=1e-4)
		self.relu = torch.nn.ReLU()

	def forward(self, x):
		return self.relu(self.bn(self.conv(x)))

class Conv(torch.nn.Module):

	def __init__(self, kernel, stride, input_dim, output_dim):
		super().__init__()
		self.fused = CONV_BN_RELU(Conv2d_tf(in_channels=input_dim, out_channels=output_dim, kernel_size=kernel, stride=stride, padding='SAME', bias=False))

	def forward(self, x):
		return self.fused(x)

class SeparableConv(torch.nn.Module):

	def __init__(self, kernel, stride, input_dim, output_dim):
		super().__init__()
		self.depthwise_conv = CONV_BN_RELU(Conv2d_tf(in_channels=input_dim, out_channels=input_dim, groups=input_dim,kernel_size=kernel, stride=stride,padding='SAME', bias=False,),)
		self.pointwise_conv = CONV_BN_RELU(Conv2d_tf(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1, padding='SAME', bias=False,),)

	def forward(self, x):
		return self.pointwise_conv(self.depthwise_conv(x))

YAMNET_PARAMS = [
	# (layer_function, kernel, stride, num_filters)
	(Conv,			[3, 3], 2,   32),
	(SeparableConv, [3, 3], 1,   64),
	(SeparableConv, [3, 3], 2,  128),
	(SeparableConv, [3, 3], 1,  128),
	(SeparableConv, [3, 3], 2,  256),
	(SeparableConv, [3, 3], 1,  256),
	(SeparableConv, [3, 3], 2,  512),
	(SeparableConv, [3, 3], 1,  512),
	(SeparableConv, [3, 3], 1,  512),
	(SeparableConv, [3, 3], 1,  512),
	(SeparableConv, [3, 3], 1,  512),
	(SeparableConv, [3, 3], 1,  512),
	(SeparableConv, [3, 3], 2, 1024),
	(SeparableConv, [3, 3], 1, 1024)
]

class YAMNet(torch.nn.Module):

	def __init__(self, num_classes=11, pretrained=False, **kwargs):
		super(YAMNet, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.pretrained = pretrained

		# Create model
		for (i, (layer_mod, kernel, stride, output_dim)) in enumerate(YAMNET_PARAMS):
			self.add_module(name='layer{}'.format(i + 1), module=layer_mod(kernel, stride, (1 if i==0 else YAMNET_PARAMS[i-1][-1]), output_dim))
		self.classifier = torch.nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)

		# Add pretrained weights on
		if self.pretrained:
			print('Using pre-trained from AudioSet')
			state_dict = torch.hub.load_state_dict_from_url(CKPT_URL, progress=False)
			del state_dict['classifier.weight']; del state_dict['classifier.bias']
			self.load_state_dict(state_dict, strict=False)

	def __str__(self):
		return self.__class__.__name__

	def forward(self, x):
		for module in self.children(): x = module(x)
		x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
		x = x.reshape(x.shape[0], -1)
		print(x.shape)
		x = self.classifier(x)
		return x


from torchinfo import summary
if __name__ == "__main__":
	model = YAMNet(num_classes=11, pretrained=True)
	summary(model, input_shape=(32,96,64,1))
