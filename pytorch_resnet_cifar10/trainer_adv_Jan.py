import argparse
import os
from pickle import TRUE
import shutil
import time
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import resnet
from utils.common import *
from datetime import datetime 
import warnings
from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
from bisect import bisect_right
import csv
import torchvision
import torchvision.transforms as transforms
from scipy.stats import truncnorm
torch.manual_seed(0)

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
	projected_gradient_descent,
)


model_names = sorted(name for name in resnet.__dict__
	if name.islower() and not name.startswith("__")
					 and name.startswith("resnet")
					 and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--prune', default='', type=str, metavar='PATH',
					help='Need pruning?')
parser.add_argument('--validation', default=0, type=int,
					help='Two step training, first step?')
parser.add_argument('--twostepfirststep', default=0, type=int,
					help='Two step training, first step?')
parser.add_argument('--twostepsecondstep', default=0, type=int,
					help='Two step training, second step?')
parser.add_argument('--CA', default='',action='store_true',
					help='Cosine Annealing?')
parser.add_argument('--dataset', default='CIFAR10', type=str,
					help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
					choices=model_names,
					help='model architecture: ' + ' | '.join(model_names) +
					' (default: resnet32)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
					help='train with channel sparsity regularization')		
parser.add_argument('--s', type=float, default=0.0001,
					help='scale sparse rate (default: 0.0001)')	
parser.add_argument('--zeta', type=float, default=0.0,
					help='pruning threshold (default: 0.0)')													
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--advtest', default=0, type=int,
					help='Alternate training of relu slope and weight')	
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
					metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--half', dest='half', action='store_true',
					help='use half-precision(16-bit) ')
parser.add_argument('--warm_up', dest='warm_up', action='store_true', help='use warm up or not')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
					help='Saves checkpoints at every specified number of epochs',
					type=int, default=10)
parser.add_argument('--epsfgsm', default=0.1, type=float,
					metavar='eps', help='FGSM attack epsilon')
parser.add_argument('--epspgd', default=0.031, type=float,
					metavar='eps', help='PGD attack epsilon')


best_prec1 = 0


def main():
	global args, best_prec1
	args = parser.parse_args()
	best_prec1 = 0.0
	if args.prune:
		args.resume = args.prune

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
	model.cuda()
	
	if not args.resume:
		with open(os.path.join(args.save_dir,'config.txt'), 'w') as args_file:
			# args_file.write(str(datetime.now())+'\n\n')
			for args_n,args_v in args.__dict__.items():
				args_v = '' if not args_v and not isinstance(args_v,int) else args_v
				args_file.write(str(args_n)+':  '+str(args_v)+'\n')

		setup_logging(os.path.join(args.save_dir, 'logger.log'))
		logging.info("saving to %s", args.save_dir)
		logging.debug("run arguments: %s", args)
	else: 
		setup_logging(os.path.join(args.save_dir, 'logger.log'), filemode='a')


	if not args.resume:
		# logging.info("creating model %s", args.arch)
		logging.info("model structure: %s", model)
		num_parameters = sum([l.nelement() for l in model.parameters()])
		logging.info("number of parameters: %d", num_parameters)


	cudnn.benchmark = True

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])



	if args.dataset == "CIFAR100":
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root='./../../data', train=True, transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			]), download=True),
			batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)

		valtest_loader = torch.utils.data.DataLoader(
			datasets.CIFAR100(root='./../../data', train=False, transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			])),
			batch_size=128, shuffle=False,
			num_workers=args.workers, pin_memory=True)
		test_loader, validation_loader = torch.utils.data.random_split(valtest_loader, [5000, 5000])
		if args.evaluate and args.validation:
			val_loader = validation_loader
		elif args.evaluate and not args.validation:
			val_loader = test_loader
		elif not args.evaluate:
			val_loader = valtest_loader
	elif args.dataset == "CIFAR10":
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(root='./../../data', train=True, transform=transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32, 4),
				transforms.ToTensor(),
				normalize,
			]), download=True),
			batch_size=args.batch_size, shuffle=True,
			num_workers=args.workers, pin_memory=True)
		valtest = datasets.CIFAR10(root='./../../data', train=False, transform=transforms.Compose([
				transforms.ToTensor(),
				normalize,
			]))
		# val_loader = torch.utils.data.DataLoader(valtest, batch_size=128, shuffle=False,
		# 												num_workers=args.workers, pin_memory=True)
		validation, test = torch.utils.data.random_split(valtest, [5000, 5000])
		if args.evaluate and args.validation:
			val_loader = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=False,
														num_workers=args.workers, pin_memory=True)
		elif args.evaluate and not args.validation:
			val_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False,
														num_workers=args.workers, pin_memory=True)
		elif not args.evaluate:
			val_loader = torch.utils.data.DataLoader(valtest, batch_size=128, shuffle=False,
														num_workers=args.workers, pin_memory=True)
		

	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda()

	if args.half:
		model.half()
		criterion.half()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	if args.arch in ['resnet1202', 'resnet110']:
		# for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
		# then switch back. In this setup it will correspond for first epoch.
		for param_group in optimizer.param_groups:
			param_group['lr'] = args.lr*0.1

	# # additional subgradient descent on the sparsity-induced penalty term
	def updateRReLU():
		for m in model.modules():	
			if isinstance(m, resnet.RReLU):
				m.a.grad.data.add_(args.s*torch.sign(m.a.data))  # L1	

		

	if args.arch in ['resnet1202', 'resnet110', 'resnet110_rotatedrelu_maam']:
		# for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
		# then switch back. In this setup it will correspond for first epoch.
		for param_group in optimizer.param_groups:
			param_group['lr'] = args.lr*0.1



	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			logging.info("=> loading checkpoint '{}'".format(args.resume))
			checkpoint_resume = torch.load(args.resume)
			args.start_epoch = checkpoint_resume['epoch']
			best_prec1 = checkpoint_resume['best_prec1']
			best_epoch = checkpoint_resume['best_epoch']
			optimizer.load_state_dict(checkpoint_resume['optimizer'])
			model.load_state_dict(checkpoint_resume['state_dict'])
			logging.info("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.evaluate, checkpoint_resume['epoch']))
		else:
			logging.info("=> no checkpoint found at '{}'".format(args.resume))
	

	if args.evaluate:		
		validate(val_loader, model, criterion)
		return


	if args.epochs == 1200 or args.epochs == 500 or args.epochs == 700:
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min = 0, last_epoch= -1, verbose=False)
	if args.epochs == 200:
		lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)
	
	
	if not args.evaluate and args.twostepfirststep:
		for name,param in model.named_parameters():
			if name[-1]!="a": #weights
				param.requires_grad = False

	if not args.evaluate and args.twostepsecondstep :
		model_dict = model.state_dict()

		if args.dataset == "CIFAR10" and args.arch == "resnet20_rotatedrelu_maam":
			pretrained_dict = torch.load("./resultsICCV/save_rrelu_twostepfirststep_resnet20_rotatedrelu_maam/checkpoint_best.th")['state_dict']	
		elif args.dataset == "CIFAR10" and args.arch == "resnet56_rotatedrelu_maam":
			pretrained_dict = torch.load("./resultsICCV/save_rrelu_twostepfirststep_resnet56_rotatedrelu_maam/checkpoint_best.th")['state_dict']	
		elif args.dataset == "CIFAR100" and args.arch == "resnet20_rotatedrelu_maam_f":
			pretrained_dict = torch.load("./resultsICCV/save_rrelu_twostepfirststep_resnet20_rotatedrelu_maam_f/checkpoint_best.th")['state_dict']	
		elif args.dataset == "CIFAR100" and args.arch == "resnet56_rotatedrelu_maam_f":
			pretrained_dict = torch.load("./resultsICCV/save_rrelu_twostepfirststep_resnet56_rotatedrelu_maam_f/checkpoint_best.th")['state_dict']	
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}	
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

		for name,param in model.named_parameters():
			if name[-1]!="a": #weights
				param.requires_grad = True
				# exit()
			else:
				param.requires_grad = False
				param.data.copy_(torch.tensor([param[i] if torch.abs(param[i])>args.zeta else 0.0 for i in range(len(param))]))
				param[torch.nonzero(param)].requires_grad = True

		
	if not args.evaluate and args.finetuning:
		model_dict_for_rrelus = model.state_dict()

		pretrained_dict = torch.load(args.finetuning)['state_dict']	
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict_for_rrelus}	
		model_dict_for_rrelus.update(pretrained_dict)
		for name, p in model.named_parameters():
			p_for_rrelu = model_dict_for_rrelus[name]
			if name[-1]=="a":
				p.requires_grad = False
				p.data.copy_(torch.tensor([p[i] if torch.abs(p_for_rrelu[i])>args.zeta else 0.0 for i in range(len(p_for_rrelu))])) #0.27 to keep same accuracy
				p[torch.nonzero(p)].requires_grad = True
						
	
	for epoch in range(args.start_epoch, args.epochs):
		if args.warm_up and epoch <5:
			for param_group in optimizer.param_groups:
				param_group['lr'] = args.lr * (epoch+1) / 5
		for param_group in optimizer.param_groups:
			logging.info('lr: %s', param_group['lr'])



		if epoch==0:
			save_checkpoint({
				'epoch': epoch +1 ,
				'best_epoch': epoch+1,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
				'optimizer': optimizer.state_dict(),
				'scheduler': lr_scheduler.state_dict()
			}, 1, filename=os.path.join(args.save_dir, 'checkpoint_init.th'))


		train(train_loader, model, criterion, optimizer, epoch, updateRReLU)
		lr_scheduler.step()

		# evaluate on validation set
		prec1, _, _ = validate(val_loader, model, criterion)

		# remember best prec
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		if is_best:
			best_epoch = epoch

		
		if is_best:
			save_checkpoint({
				'epoch': epoch +1 ,
				'best_epoch': best_epoch,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
				'optimizer': optimizer.state_dict(),
				'scheduler': lr_scheduler.state_dict()
			}, is_best, filename=os.path.join(args.save_dir, 'checkpoint_best.th'))
				

		logging.info('\n Epoch: {0}\t'
					 'Val_Prec1 {prec1:.4f} \t'
					 'Best Validation Prec@1 {best_prec1:.3f} \t'
					 .format(epoch + 1, prec1=prec1, best_prec1=best_prec1))
	
	logging.info('*'*50+'DONE'+'*'*50)
	logging.info('\n Best_Epoch: {0}\t'
					 'Best_Prec1 {best_prec1:.4f} \t'
					 .format(best_epoch+1,  best_prec1=best_prec1))


def train(train_loader, model, criterion, optimizer, epoch, updateRReLU):
	"""
		Run one train epoch
	"""
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()	

	
	# switch to train mode
	model.train()


	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda()
		input_var = input.cuda()
		target_var = target
		if args.half:
			input_var = input_var.half()

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)#+0.1*regularizer

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		if args.sr:
			updateRReLU()
		optimizer.step()
					

		output = output.float()
		loss = loss.float()
		# measure accuracy and record loss
		prec1 = accuracy(output.data, target)[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			logging.info('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
	"""
	Run evaluation
	"""
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
		list_rreluscaling=[]
		for name, p in model.named_parameters():
			if name[-1]=="a":
				p.data.copy_(torch.tensor([p[i] if torch.abs(p[i])>args.zeta else 0.0 for i in range(len(p))])) #0.27 to keep same accuracy
				list_rreluscaling.append(p.nonzero().size(0))		
				zeros += p.numel() - p.nonzero().size(0)
				slopes += p.numel()
	
		print("Filters still alive along the depth: ", list_rreluscaling)
		print("Total number of filters that can be ignored: ", zeros)
		print("Total number of filters: ", slopes)

		# zeros = 0.0
		# slopes = 0.0
		# list_bnscaling = []
		# for name, p in model.named_parameters():
		# 	# print(name[-8:], p.shape)
		# 	if name[-10:]=="bn1.weight" or name[-10:]=="bn2.weight" or name[-10:]=="bn3.weight":
		# 	# if name[-8:]=="rrelu1.a":
	
		# 		p.data.copy_(torch.tensor([p[i] if torch.abs(p[i])>args.zeta else 0.0 for i in range(len(p))])) #0.27 to keep same accuracy
		# 		list_bnscaling.append(p.nonzero().size(0))		
		# 		zeros += p.numel() - p.nonzero().size(0)
		# 		slopes += p.numel()
	
		# print(len(list_bnscaling), list_bnscaling)
		# print(zeros, slopes)

		# zeros = 0.0
		# slopes = 0.0
		# list_ = []
		# param_rots = []
		# param_bns = []
		# for name, p in model.named_parameters():
		# 	if name[-1]=="a":
		# 		param_rots.append(p)
		# 	if name[-10:]=="bn1.weight" or name[-10:]=="bn2.weight" or name[-10:]=="bn3.weight":
		# 		param_bns.append(p)
		# for j in range(len(param_rots)):
		# 	param_rots[j] = torch.tensor([param_rots[j][i] if (torch.abs(param_rots[j][i])>args.zeta and torch.abs(param_bns[j][i])>args.zeta) else 0.0 for i in range(len(param_rots[j]))])
		# 	param_bns[j] = torch.tensor([param_bns[j][i] if (torch.abs(param_rots[j][i])>args.zeta and torch.abs(param_bns[j][i])>args.zeta) else 0.0 for i in range(len(param_bns[j]))])			
		# 	list_.append(torch.mul(param_rots[j],param_bns[j]).nonzero().size(0))		
		# 	zeros += torch.mul(param_rots[j],param_bns[j]).numel() - torch.mul(param_rots[j],param_bns[j]).nonzero().size(0)
		# 	slopes += torch.mul(param_rots[j],param_bns[j]).numel()
		
		# for name, p in model.named_parameters():
		# 	if name[-1]=="a":
		# 		p.data.copy_(param_rots.pop(0))
		# 	if name[-10:]=="bn1.weight" or name[-10:]=="bn2.weight" or name[-10:]=="bn3.weight":
		# 		p.data.copy_(param_bns.pop(0))
		# print(len(list_), list_)
		# print(zeros, slopes)

		# print(sum([1 if list_rreluscaling[i]==list_bnscaling[i] else 0 for i in range(len(list_bnscaling)) ]))
		


	## the distribution of relus
	A = []
	for name, p in model.named_parameters():
		if name[-1]=="a" or name[-1]=="c":
			A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
	plt.figure(figsize=(3.5,2.5))
	plt.grid()
	plt.hist(A, bins=50)
	plt.savefig("hist_rotationscaling_rrelu")


	## the distribution of BN scaling parameters
	A = []
	for name, p in model.named_parameters():
		if name[-10:]=="bn1.weight" or name[-10:]=="bn2.weight" or name[-10:]=="bn3.weight":
			A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
	plt.figure(figsize=(3.5,2.5))
	plt.grid()
	plt.hist(A, bins=50)
	plt.savefig("hist_BNscaling_rrelu")
	

	if args.advtest:
		length=0.0
		total_loss_fgsm = 0.0
		for i, (input, target) in enumerate(val_loader):
			length += 1.0
			target = target.cuda()
			input_var = input.cuda()
			target_var = target.cuda()


			if args.half:
				input_var = input_var.half()
						
			total_loss_fgsm += estimate_llc(model, input_var, args.epsfgsm)

		LLC = total_loss_fgsm/(length*input_var.size(0))
		print("Empirical LLC: ", LLC)

	


	# with torch.no_grad():
	for i, (input, target) in enumerate(val_loader):

		target = target.cuda()
		input_var = input.cuda()
		target_var = target.cuda()


		if args.half:
			input_var = input_var.half()
					
		# compute output
		output = model(input_var)		
		loss = criterion(output, target_var)#+0.1*regularizer
		output = output.float()
		loss = loss.float()
		# measure accuracy and record loss
		prec1 = accuracy(output.data, target)[0]
		losses.update(loss.item(), input.size(0))
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
						i, len(val_loader), batch_time=batch_time, loss=losses,top1=top1,
						loss_pgd=losses_pgd, top1_pgd=top1_pgd, loss_fgsm=losses_fgsm, top1_fgsm=top1_fgsm))
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


	if args.prune:
		save_checkpoint({
			'state_dict': model.state_dict(),
			'best_prec1': top1.avg,
		},True, filename=os.path.join(args.save_dir, 'checkpoint_pruned.th'))
	
	return top1.avg, top1_pgd.avg, top1_fgsm.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar' ):
	"""
	Save the training model
	"""
	torch.save(state, filename)
	

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
