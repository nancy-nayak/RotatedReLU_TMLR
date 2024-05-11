# python train.py --dataset cifar100 --layers 40 --widen-factor 4
import argparse
import os
import shutil
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
from wideresnet import WideResNet
import matplotlib.pyplot as plt


from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
	projected_gradient_descent,
)

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

def setup_logging(log_file='log.txt',filemode='w'):
	"""Setup logging configuration
	"""
	logging.basicConfig(level=logging.DEBUG,
						format="%(asctime)s - %(levelname)s - %(message)s",
						datefmt="%Y-%m-%d %H:%M:%S",
						filename=log_file,
						filemode=filemode)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--finetuning', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=200, type=int,
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
					help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
					help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
					help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
					help='widen factor (default: 10)')
parser.add_argument('--retrain', default=0, type=int,
					help='retrained on trained model (default: 0)')
parser.add_argument('--droprate', default=0, type=float,
					help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
					help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('--gamma', type=float, default=0.0,
					help='pruning threshold (default: 0.0)')
parser.add_argument('--prune', default='', type=str,
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
					help='name of experiment')
parser.add_argument('--typer', default='ReLU', type=str,
					help='activation')
parser.add_argument('--advtest', default=0, type=int,
					help='Alternate training of relu slope and weight')
parser.add_argument('--epsfgsm', default=0.1, type=float,
					metavar='eps', help='FGSM attack epsilon')
parser.add_argument('--epspgd', default=0.031, type=float,
					metavar='eps', help='PGD attack epsilon')
parser.add_argument('--tensorboard',
					help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='save_temp', type=str)
parser.add_argument('--warm_up', dest='warm_up', action='store_true', help='use warm up or not')

best_prec1 = 0
def local_lip(model, x, xp, top_norm, btm_norm, reduction='mean'):
	model.eval()
	down = torch.flatten(x - xp, start_dim=1)
	if top_norm == "kl":
		criterion_kl = nn.KLDivLoss(reduction='none')
		top = criterion_kl(F.log_softmax(model(xp), dim=1),
						   F.softmax(model(x), dim=1))
		ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
	else:
		top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
		ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

	if reduction == 'mean':
		return torch.mean(ret)
	elif reduction == 'sum':
		return torch.sum(ret)
	else:
		raise ValueError(f"Not supported reduction: {reduction}")

def preprocess_x(x):
	return torch.from_numpy(x.transpose(0, 3, 1, 2)).float()


def estimate_llc(model, input_var, epsilon):
	device="cuda"
	x=input_var
	step_size=0.001
	top_norm="kl"
	btm_norm=np.inf
	perturb_steps=60

	# generate adversarial example
	if btm_norm in [1, 2, np.inf]:
		x_adv = x + 0.001 * torch.randn(x.shape).to(device)

		# Setup optimizers
		optimizer = torch.optim.SGD([x_adv], lr=step_size)

		for _ in range(perturb_steps):
			x_adv.requires_grad_(True)
			optimizer.zero_grad()
			with torch.enable_grad():
				loss = (-1) * local_lip(model, x, x_adv, top_norm, btm_norm)
			loss.backward()
			# renorming gradient
			eta = step_size * x_adv.grad.data.sign().detach()
			x_adv = x_adv.data.detach() + eta.detach()
			eta = torch.clamp(x_adv.data - x.data, - epsilon, epsilon)
			x_adv = x.data.detach() + eta.detach()
			x_adv = torch.clamp(x_adv, 0.0, 1.0)
	else:
		raise ValueError(f"Unsupported norm {btm_norm}")

	loss = local_lip(model, x, x_adv, top_norm, btm_norm, reduction='sum').item()
	
	return loss

def main():
	global args, best_prec1
	args = parser.parse_args()
	if args.tensorboard: configure("runs/%s"%(args.name))

	# Check the save_dir exists or not
	if not os.path.exists(os.path.join(args.save_dir, "%s/"%(args.name), "%s/"%(args.typer))):
		os.makedirs(os.path.join(args.save_dir, "%s/"%(args.name), "%s/"%(args.typer)))

	# Data loading code
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
									 std=[x/255.0 for x in [63.0, 62.1, 66.7]])

	if args.augment:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
								(4,4,4,4),mode='reflect').squeeze()),
			transforms.ToPILImage(),
			transforms.RandomCrop(32),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
			])
	else:
		transform_train = transforms.Compose([
			transforms.ToTensor(),
			normalize,
			])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize
		])

	kwargs = {'num_workers': 1, 'pin_memory': True}
	assert(args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'SVHN')
	if not args.dataset == 'SVHN':
		train_loader = torch.utils.data.DataLoader(
			datasets.__dict__[args.dataset.upper()]('../data', train=True, download=True,
							transform=transform_train),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = torch.utils.data.DataLoader(
			datasets.__dict__[args.dataset.upper()]('../data', train=False, transform=transform_test),
			batch_size=args.batch_size, shuffle=True, **kwargs)
	else:
		train_loader = torch.utils.data.DataLoader(
			datasets.__dict__[args.dataset.upper()]('../data', split='train',  download=True,
							transform=transform_train),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = torch.utils.data.DataLoader(
			datasets.__dict__[args.dataset.upper()]('../data',  split='test', download=True,
							transform=transform_test),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		
	# create model
	if not args.dataset == 'SVHN':
		model = WideResNet(args.layers, args.typer, args.dataset == 'cifar10' and 10 or 100,
							args.widen_factor, dropRate=args.droprate)
	if args.dataset == 'SVHN':
		model = WideResNet(args.layers, args.typer, 10,
								args.widen_factor, dropRate=args.droprate)

	# get the number of model parameters
	print('Number of model parameters: {}'.format(
		sum([p.data.nelement() for p in model.parameters()])))

	# for training on multiple GPUs.
	# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
	# model = torch.nn.DataParallel(model).cuda()
	model = model.cuda()

	setup_logging(os.path.join(args.save_dir, "%s/"%(args.name), "%s/"%(args.typer), 'logger.log'))
	logging.info("model structure: %s", model)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	if args.prune:
		if os.path.isfile(args.prune):
			logging.info("=> loading checkpoint '{}'".format(args.prune))
			checkpoint = torch.load(args.prune)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			logging.info("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.evaluate, checkpoint['epoch']))
		else:
			logging.info("=> no checkpoint found at '{}'".format(args.prune))

	if args.retrain:
		model_dict = model.state_dict()
		checkpoint_resume = torch.load("./runs/WideResNet-40-4/ReLU/model_best.pth.tar")
		pretrained_dict = checkpoint_resume['state_dict']	
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}	
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

		for name,param in model.named_parameters():
			if name[-1]=="a" or name[-1]=="c":
				param.data.copy_(torch.tensor([1.0 for _ in range(len(param))])) 
				# param.data.copy_(torch.tensor(truncnorm.rvs( (np.tan(40.0/180*np.pi)-1.0)/np.sqrt(0.5), (np.tan(50.0/180*np.pi)-1.0)/np.sqrt(0.5), size=len(param))*np.sqrt(0.5)+1.0))
				param.requires_grad = True
			else:
				param.requires_grad = True

	cudnn.benchmark = True

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum, nesterov = args.nesterov,
								weight_decay=args.weight_decay)

	# cosine learning rate
	if not args.retrain:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
	else:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
		# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 250], gamma=0.01, last_epoch=args.start_epoch - 1)
	
	if not args.evaluate and args.finetuning:
		model_dict_for_rrelus = model.state_dict()

		pretrained_dict = torch.load(args.finetuning)['state_dict']	
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict_for_rrelus}	
		model_dict_for_rrelus.update(pretrained_dict)
		for name, p in model.named_parameters():
			p_for_rrelu = model_dict_for_rrelus[name]
			if name[-1]=="a" or name[-1]=="c":
				p.requires_grad = False
				p.data.copy_(torch.tensor([p[i] if torch.abs(p_for_rrelu[i])>args.gamma else 0.0 for i in range(len(p_for_rrelu))])) #0.27 to keep same accuracy
				p[torch.nonzero(p)].requires_grad = True

	if args.evaluate:
		validate(val_loader, model, criterion, 0)
		return

	for epoch in range(args.start_epoch, args.epochs):
		if args.warm_up and epoch <5:
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr * (epoch+1) / 5
		for param_group in optimizer.param_groups:
			logging.info('lr: %s', param_group['lr'])

	for epoch in range(args.start_epoch, args.epochs):
		# train for one epoch
		train(train_loader, model, criterion, optimizer, scheduler, epoch)

		# evaluate on validation set
		prec1 = validate(val_loader, model, criterion, epoch)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
		}, is_best)
		logging.info('Epoch {} \t current prec {} \t* Prec@1 {}'.format(epoch, prec1, best_prec1))
	print('Best accuracy: ', best_prec1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
	"""Train for one epoch on the training set"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		target = target.cuda(non_blocking=True)
		input = input.cuda(non_blocking=True)

		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			logging.info('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  loss=losses, top1=top1))

	# log to TensorBoard
	if args.tensorboard:
		log_value('train_loss', losses.avg, epoch)
		log_value('train_acc', top1.avg, epoch)

def validate(val_loader, model, criterion, epoch):
	"""Perform validation on the validation set"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	losses_pgd = AverageMeter()
	top1_pgd = AverageMeter()
	losses_fgsm = AverageMeter()
	top1_fgsm = AverageMeter()

	# switch to evaluate mode
	model.eval()
	end = time.time()

	if args.prune:
		## Simple pruning
		zeros = 0.0
		slopes = 0.0
		list1=[]
		for name, p in model.named_parameters():
			# print(name, p.shape)
			if name[-1]=="a" or name[-1]=="c":
				# med = torch.median(p)
				max_ = torch.max(torch.abs(p))
				# p.data.copy_(torch.tensor([i if torch.abs(i)>0.72*torch.abs(med) else 0.0 for i in p]))
				# p.data.copy_(torch.tensor([i if torch.abs(i)>0.45*max_ else 0.0 for i in p])) #0.27 to keep same accuracy
				p.data.copy_(torch.tensor([p[i] if torch.abs(p[i])>args.gamma else 0.0 for i in range(len(p))])) #0.27 to keep same accuracy
				list1.append(p.nonzero().size(0))
				# exit()
				# if len(np.nonzero(abs(p.cpu().detach().numpy())<0.04)[0])>0:
				# 	print(name, "number of zero relu slope......", len(np.nonzero(abs(p.cpu().detach().numpy())<0.04)[0]))
				# print(np.nonzero(abs(p.cpu().detach().numpy())<0.04)[0], len(p))
				zeros += p.numel() - p.nonzero().size(0)
				slopes += p.numel()
				# print(max_, p)
		print(len(list1), list1)
		print(zeros, slopes)
	# exit()
	
	# ## the values of relus
	# A = []
	# for name, p in model.named_parameters():
	# 	if name[-1]=="a" or name[-1]=="c":
	# 		print(name)
	# 		A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
	# print(len(A))
	# plt.figure(figsize=(3.5,2.5))
	# plt.hist(A, bins=100)
	# plt.savefig("hist_wideresnet404_f")
	# exit()

	if args.advtest:
		length=0.0
		total_loss_fgsm = 0.0
		for i, (input, target) in enumerate(val_loader):
			length += 1.0
			target = target.cuda()
			input_var = input.cuda()
			target_var = target.cuda()


			# if args.half:
			# 	input_var = input_var.half()
						
			total_loss_fgsm += estimate_llc(model, input_var, args.epsfgsm)

		LLC = total_loss_fgsm/(length*input_var.size(0))
		print(LLC)

	for i, (input, target) in enumerate(val_loader):
		target = target.cuda(non_blocking=True)
		input_var = input.cuda(non_blocking=True)
		target_var = target.cuda()

		# compute output
		with torch.no_grad():
			output = model(input_var)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target, topk=(1,))[0]
		losses.update(loss.data.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))


		if args.prune and args.advtest:
			input_var_pgd = projected_gradient_descent(model, input_var, args.epspgd, args.epspgd/5, 60, np.inf)
			# compute output for pgd samles
			output_pgd = model(input_var_pgd)
			loss_pgd = criterion(output_pgd, target_var)
			output_pgd = output_pgd.float()
			loss_pgd = loss_pgd.float()

			# measure accuracy and record loss
			prec1_pgd = accuracy(output_pgd.data, target)[0]
			losses_pgd.update(loss_pgd.item(), input.size(0))
			top1_pgd.update(prec1_pgd.item(), input.size(0))

			
			input_var_fgsm = fast_gradient_method(model, input_var, args.epsfgsm, np.inf)
			# compute output for fgsm samles
			output_fgsm = model(input_var_fgsm)
			loss_fgsm = criterion(output_fgsm, target_var)
			output_fgsm = output_fgsm.float()
			loss_fgsm = loss_fgsm.float()
			
			# measure accuracy and record loss
			prec1_fgsm = accuracy(output_fgsm.data, target)[0]
			losses_fgsm.update(loss_fgsm.item(), input.size(0))
			top1_fgsm.update(prec1_fgsm.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if args.prune and args.advtest:
			if i % args.print_freq == 0:
				logging.info('Test: [{0}/{1}]\t'
							'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
							'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
							'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
							'Loss PGD {loss_pgd.val:.4f} ({loss_pgd.avg:.4f})\t'
							'Prec@1 PGD {top1_pgd.val:.3f} ({top1_pgd.avg:.3f})'
							'Loss FGSM {loss_fgsm.val:.4f} ({loss_fgsm.avg:.4f})\t'
							'Prec@1 FGSM {top1_fgsm.val:.3f} ({top1_fgsm.avg:.3f})'.format(
						i, len(val_loader), batch_time=batch_time, loss=losses,
						top1=top1, loss_pgd=losses_pgd, top1_pgd=top1_pgd, loss_fgsm=losses_fgsm, top1_fgsm=top1_fgsm))
		if i % args.print_freq == 0:
			logging.info('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1))

	if args.prune and args.advtest:
		logging.info(' * Prec@1 {top1.avg:.3f}\t'' * Prec@1 PGD {top1_pgd.avg:.3f}\t'' * Prec@1 FGSM {top1_fgsm.avg:.3f}\t'
					' *LLC FGSM {LLC}\t'.format(top1=top1, top1_pgd=top1_pgd, top1_fgsm=top1_fgsm, LLC=LLC))
	logging.info(' * Prec@1 {top1.avg:.3f}\t'.format(top1=top1))
	
	# log to TensorBoard
	if args.tensorboard:
		log_value('val_loss', losses.avg, epoch)
		log_value('val_acc', top1.avg, epoch)
	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	"""Saves checkpoint to disk"""
	directory = args.save_dir + "/%s/"%(args.name)+ "%s/"%(args.typer)
	if not os.path.exists(directory):
		os.makedirs(directory)
	filename = directory + filename
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, args.save_dir + '/%s/'%(args.name) + '%s/'%(args.typer) + 'model_best.pth.tar')

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

if __name__ == '__main__':
	main()
