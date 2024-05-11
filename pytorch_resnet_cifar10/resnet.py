'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *
from scipy.stats import truncnorm
import numpy as np
from torch.autograd import Variable
from channel_selection import channel_selection


__all__ = ['resnet20', 'resnet56', 'resnet20_f', 'resnet56_f', 'resnet110_identitymapping', 'resnet110_identitymapping_f', 'resnet110_rotatedrelu_maam_identitymapping_a', 'resnet110_rotatedrelu_maam_identitymapping_a_f', 'resnet164', 'resnet164_f', 'resnet164_rotatedrelu_maam', 'resnet164_rotatedrelu_maam_f']


def _weights_init(m):
	classname = m.__class__.__name__
	#print(classname)
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

class RReLU(nn.Module):
	def __init__(self, in_planes):
		super().__init__()
		aa = np.random.uniform(0.0,1.0,in_planes)
		self.a = nn.Parameter(torch.tensor((aa>0.5)*(truncnorm.rvs( (np.tan(35.0/180*np.pi)-1.0)/np.sqrt(3.0), (np.tan(55.0/180*np.pi)-1.0)/np.sqrt(3.0), size=in_planes)*np.sqrt(3.0)+1.0) \
							+ (aa<=0.5)*(truncnorm.rvs((np.tan(-55.0/180*np.pi)+1.0)/np.sqrt(3.0), (np.tan(-35.0/180*np.pi)+1.0)/np.sqrt(3.0), size=in_planes)*np.sqrt(3.0)-1.0)).float(),requires_grad=True)

	def forward(self, x):
		temp = x
		# a1 = torch.repeat_interleave(torch.repeat_interleave(self.a.repeat(temp.shape[0],1).unsqueeze(2), temp.shape[2], dim=2).unsqueeze(3), temp.shape[3],dim=3)	
		# out = torch.mul(a1,torch.relu(temp))
		out = torch.mul(self.a[None, :, None, None],torch.relu(temp))

		return out

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out

class BasicBlock_identitymapping(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock_identitymapping, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.relu1 = nn.ReLU(inplace=True)
		self.relu2 = nn.ReLU(inplace=True)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = self.conv1(self.relu1(self.bn1(x)))
		out = self.conv2(self.relu2(self.bn2(out)))
		out = out + self.shortcut(x)
		return out

class BasicBlock_rotatedrelu_maam(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock_rotatedrelu_maam, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.rrelu1 = RReLU(planes)
		self.rrelu2 = RReLU(planes)
		self.dropout = nn.Dropout(0.2)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = self.rrelu1(self.bn1(self.conv1(x)))
		
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		
		out = self.rrelu2(out)	
		
		return out

class BasicBlock_rotatedrelu_maam_identitymapping_a(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock_rotatedrelu_maam_identitymapping_a, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.rrelu1 = RReLU(in_planes)
		self.rrelu2 = RReLU(planes)		

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):

		temp = self.conv1(self.rrelu1( self.bn1(x)))				
		out = self.conv2(self.rrelu2(self.bn2(temp)))
		out1 = out + self.shortcut(x)
				
		return out1


"""
preactivation resnet with bottleneck design.
"""

class BasicblockBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
		super(BasicblockBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.select = channel_selection(inplanes)
		self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(cfg[1])
		self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(cfg[2])
		self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.bn1(x)
		out = self.select(out)
		out = self.relu(out)
		
		out = self.conv1(out)
		# print("Inside resnet block, after conv1", out.shape)

		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)
		# print("Inside resnet block, after conv2", out.shape)

		out = self.bn3(out)
		out = self.relu(out)
		out = self.conv3(out)
		# print("Inside resnet block, after conv3", out.shape)


		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual

		return out

class BasicblockBottleneck_rotated(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
		super(BasicblockBottleneck_rotated, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.select = channel_selection(inplanes)
		self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(cfg[1])
		self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(cfg[2])
		self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.rrelu1 = RReLU(inplanes)
		self.rrelu2 = RReLU(cfg[1])
		self.rrelu3 = RReLU(cfg[2])
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.bn1(x)
		out = self.select(out)
		out = self.rrelu1(out)
		
		out = self.conv1(out)

		out = self.bn2(out)
		out = self.rrelu2(out)
		out = self.conv2(out)

		out = self.bn3(out)
		out = self.rrelu3(out)
		out = self.conv3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual

		return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)
		# self.softmax = nn.Softmax(dim=1)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))		
		out = self.layer1(out)		
		out = self.layer2(out)		
		out = self.layer3(out)	
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


class ResNet_f(nn.Module):
	def __init__(self, block, num_blocks, num_classes=100):
		super(ResNet_f, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		
		out = self.layer1(out)
		
		out = self.layer2(out)
		
		out = self.layer3(out)
		
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

class ResNet_rotated(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet_rotated, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)
		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)	
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

class ResNet_identitymapping(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet_identitymapping, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.linear = nn.Linear(64, num_classes)
		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)	
		out = self.bn2(out)	
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out
	
class ResNet_identitymapping_f(nn.Module):
	def __init__(self, block, num_blocks, num_classes=100):
		super(ResNet_identitymapping_f, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.linear = nn.Linear(64, num_classes)
		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)	
		out = self.bn2(out)	
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


class ResNet_rotated_f(nn.Module):
	def __init__(self, block, num_blocks, num_classes=100):
		super(ResNet_rotated_f, self).__init__()
		self.in_planes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		temp1 = self.bn1(self.conv1(x))
		out = F.relu(temp1)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)		
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out

class ResNetBottleneck(nn.Module):
	def __init__(self, depth=164, dataset='cifar10', cfg=None):
		super(ResNetBottleneck, self).__init__()
		assert int((depth - 2) % 9) == 0, 'depth should be 9n+2'

		n = (depth - 2) // 9
		block = BasicblockBottleneck

		if cfg is None:
			# Construct config variable.
			cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
			cfg = [item for sub_list in cfg for item in sub_list]

		self.inplanes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
							   bias=False)
		self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
		self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
		self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
		self.bn = nn.BatchNorm2d(64 * block.expansion)
		self.select = channel_selection(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)

		if dataset == 'cifar10':
			self.fc = nn.Linear(cfg[-1], 10)
		elif dataset == 'cifar100':
			self.fc = nn.Linear(cfg[-1], 100)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(0.5)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, cfg, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)  # 32x32
		x = self.layer2(x)  # 16x16
		x = self.layer3(x)  # 8x8
		x = self.bn(x)
		x = self.select(x)
		x = self.relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

class ResNetBottleneck_f(nn.Module):
	def __init__(self, depth=164, dataset='cifar100', cfg=None):
		super(ResNetBottleneck_f, self).__init__()
		assert int((depth - 2) % 9) == 0, 'depth should be 9n+2'

		n = (depth - 2) // 9
		block = BasicblockBottleneck

		if cfg is None:
			# Construct config variable.
			cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
			cfg = [item for sub_list in cfg for item in sub_list]

		self.inplanes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
							   bias=False)
		self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
		self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
		self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
		self.bn = nn.BatchNorm2d(64 * block.expansion)
		self.select = channel_selection(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)

		if dataset == 'cifar10':
			self.fc = nn.Linear(cfg[-1], 10)
		elif dataset == 'cifar100':
			self.fc = nn.Linear(cfg[-1], 100)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(0.5)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, cfg, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.layer1(x)  # 32x32
		x = self.layer2(x)  # 16x16
		x = self.layer3(x)  # 8x8
		x = self.bn(x)
		x = self.select(x)
		x = self.relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
	
class ResNetBottleneck_rotated(nn.Module):
	def __init__(self, depth=164, dataset='cifar10', cfg=None):
		super(ResNetBottleneck_rotated, self).__init__()
		assert int((depth - 2) % 9) == 0, 'depth should be 9n+2'

		n = (depth - 2) // 9
		block = BasicblockBottleneck_rotated

		if cfg is None:
			# Construct config variable.
			cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
			cfg = [item for sub_list in cfg for item in sub_list]

		self.inplanes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
							   bias=False)
		self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
		self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
		self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
		self.bn = nn.BatchNorm2d(64 * block.expansion)
		self.select = channel_selection(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)

		if dataset == 'cifar10':
			self.fc = nn.Linear(cfg[-1], 10)
		elif dataset == 'cifar100':
			self.fc = nn.Linear(cfg[-1], 100)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(0.5)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, cfg, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)

		x = self.layer1(x)  # 32x32
		x = self.layer2(x)  # 16x16
		x = self.layer3(x)  # 8x8
		x = self.bn(x)
		x = self.select(x)
		x = self.relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
	
class ResNetBottleneck_rotated_f(nn.Module):
	def __init__(self, depth=164, dataset='cifar100', cfg=None):
		super(ResNetBottleneck_rotated_f, self).__init__()
		assert int((depth - 2) % 9) == 0, 'depth should be 9n+2'

		n = (depth - 2) // 9
		block = BasicblockBottleneck_rotated

		if cfg is None:
			# Construct config variable.
			cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
			cfg = [item for sub_list in cfg for item in sub_list]

		self.inplanes = 16

		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
							   bias=False)
		self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
		self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
		self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
		self.bn = nn.BatchNorm2d(64 * block.expansion)
		self.select = channel_selection(64 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.avgpool = nn.AvgPool2d(8)

		if dataset == 'cifar10':
			self.fc = nn.Linear(cfg[-1], 10)
		elif dataset == 'cifar100':
			self.fc = nn.Linear(cfg[-1], 100)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(0.5)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, cfg, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
			)

		layers = []
		layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)

		x = self.layer1(x)  # 32x32
		x = self.layer2(x)  # 16x16
		x = self.layer3(x)  # 8x8
		x = self.bn(x)
		x = self.select(x)
		x = self.relu(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
	
#### ResNet20 versions =========================================================
## CIFAR10  ====================================================================
def resnet20():
	return ResNet(BasicBlock, [3, 3, 3])
		
def resnet20_rotatedrelu_maam():
	return ResNet_rotated(BasicBlock_rotatedrelu_maam, [3, 3, 3])

## CIFAR100 ====================================================================
def resnet20_f():
	return ResNet_f(BasicBlock, [3, 3, 3])

def resnet20_rotatedrelu_maam_f():
	return ResNet_rotated_f(BasicBlock_rotatedrelu_maam, [3, 3, 3])

##==============================================================================


#### ResNet56 versions =========================================================
## CIFAR10  ====================================================================
def resnet56():
	return ResNet(BasicBlock, [9, 9, 9])

def resnet56_rotatedrelu_maam():
	return ResNet_rotated(BasicBlock_rotatedrelu_maam, [9, 9, 9])

## CIFAR100 ====================================================================
def resnet56_f():
	return ResNet_f(BasicBlock, [9, 9, 9])

def resnet56_rotatedrelu_maam_f():
	return ResNet_rotated_f(BasicBlock_rotatedrelu_maam, [9, 9, 9])

##==============================================================================

#### ResNet110 versions =========================================================
## CIFAR10  ====================================================================
def resnet110_identitymapping():
	return ResNet_identitymapping(BasicBlock_identitymapping, [18, 18, 18])

def resnet110_rotatedrelu_maam_identitymapping_a():## one using
	return ResNet_identitymapping(BasicBlock_rotatedrelu_maam_identitymapping_a, [18, 18, 18])

## CIFAR100 ====================================================================
def resnet110_identitymapping_f():
	return ResNet_identitymapping_f(BasicBlock_identitymapping, [18, 18, 18])

def resnet110_rotatedrelu_maam_identitymapping_a_f():
	return ResNet_identitymapping_f(BasicBlock_rotatedrelu_maam_identitymapping_a, [18, 18, 18])


### ResNet164 versions =========================================================
def resnet164():
	return ResNetBottleneck()

def resnet164_f():
	return ResNetBottleneck_f()

def resnet164_rotatedrelu_maam():
	return ResNetBottleneck_rotated()

def resnet164_rotatedrelu_maam_f():
	return ResNetBottleneck_rotated_f()


def test(net):
	total_params = 0

	for x in filter(lambda p: p.requires_grad, net.parameters()):
		total_params += np.prod(x.data.numpy().shape)
	print("Total number of params", total_params)
	print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
	for net_name in __all__:
		if net_name.startswith('resnet'):
			print(net_name)
			test(globals()[net_name]())
			print()