from models.resnet20 import ResNet as resnet_model
from models.mobilenetv2 import MobileNetV2 as mobilenetv2_model
from models.yamnet import YAMNet as yamnet_model

def get_model_fn(name='resnet20'):
	if name=='resnet20':
		return resnet_model
	elif name=='mobilenet':
		return mobilenetv2_model
	elif name=='yamnet':
		return yamnet_model
	else:
		raise ValueError(f"Unknown model name: {name}")
