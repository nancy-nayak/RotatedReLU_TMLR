from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import logging
from utils.common import *
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
					help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
					help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
					help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
					help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
					help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
					help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
					help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
					help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
					help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
					help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
					help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
					help='depth of the neural network')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--prune', default='', type=str, metavar='PATH',
					help='Need pruning?')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
	os.makedirs(args.save)




kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data.cifar10', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.Pad(4),
						   transforms.RandomCrop(32),
						   transforms.RandomHorizontalFlip(),
						   transforms.ToTensor(),
						   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR100('./data.cifar100', train=True, download=True,
					   transform=transforms.Compose([
						   transforms.Pad(4),
						   transforms.RandomCrop(32),
						   transforms.RandomHorizontalFlip(),
						   transforms.ToTensor(),
						   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
					   ])),
		batch_size=args.batch_size, shuffle=True, **kwargs)
	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
					   ])),
		batch_size=args.test_batch_size, shuffle=True, **kwargs)

n = (args.depth - 2) // 9
cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
cfg = [item for sub_list in cfg for item in sub_list]

if args.refine:
	checkpoint = torch.load(args.refine)
	model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
	model.load_state_dict(checkpoint['state_dict'])
elif args.prune:
	checkpoint = torch.load(args.prune)
	# model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
	model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=cfg)
	model.load_state_dict(checkpoint['state_dict'])
else:
	model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
	model.cuda()

setup_logging(os.path.join(args.save, 'logger.log'))
logging.info("saving to %s", args.save)
logging.debug("run arguments: %s", args)
num_parameters = sum([l.nelement() for l in model.parameters()])
logging.info("number of parameters: %d", num_parameters)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
warm_up = True
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-warm_up*4, eta_min = 0, last_epoch= -1, verbose=False)

if args.resume:
	if os.path.isfile(args.resume):
		print("=> loading checkpoint '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
			  .format(args.resume, checkpoint['epoch'], best_prec1))
	else:
		print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
	for m in model.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		pred = output.data.max(1, keepdim=True)[1]
		loss.backward()
		if args.sr:
			updateBN()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			logging.info('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))

def test():
	model.eval()

	# ## the distribution of relus
	# A = []
	# for name, p in model.named_parameters():
	# 	if name[-1]=="a" or name[-1]=="c":
	# 		# print(name)
	# 		A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
	# print(len(A))
	# plt.figure(figsize=(3.5,2.5))
	# plt.hist(A, bins=20)
	# plt.savefig("hist_rotationscling_resnet164rrelu_woreg")

	## the distribution of BN scaling parameters
	A = []
	for name, p in model.named_parameters():
		if name[-10:]=="bn1.weight" or name[-10:]=="bn2.weight" or name[-10:]=="bn3.weight":
		# if name[-14:]=="2.6.bn1.weight" or name[-14:]=="2.6.bn2.weight" or name[-14:]=="2.6.bn3.weight":
			# print(name, name[7:15])
			# exit()
			A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
	print(len(A))
	plt.figure(figsize=(3.5,2.5))
	plt.hist(A, bins=20)
	plt.savefig("hist_BNscaling_resnet164_woreg_netslim_lrscheduleradded")

	exit()


	test_loss = 0
	correct = 0
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
	torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
	if is_best:
		shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if args.evaluate:		
	test()
	exit()



best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
	if epoch in [args.epochs*0.5, args.epochs*0.75]:
		for param_group in optimizer.param_groups:
			param_group['lr'] *= 0.1
	train(epoch)
	lr_scheduler.step()
	prec1 = test()
	is_best = prec1 > best_prec1
	best_prec1 = max(prec1, best_prec1)
	save_checkpoint({
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_prec1': best_prec1,
		'optimizer': optimizer.state_dict(),
	}, is_best, filepath=args.save)

logging.info("Best accuracy: "+str(best_prec1))