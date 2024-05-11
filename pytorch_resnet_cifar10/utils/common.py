import os
import torch
import logging.config
import shutil
import torch.nn as nn
import numpy
import datetime
import torch.utils.data as data_utils
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

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


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
	filename = os.path.join(path, filename)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
	if save_all:
		shutil.copyfile(filename, os.path.join(
			path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


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

	_, pred = output.float().topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def get_time(delta,epoch,epochs):
	now = datetime.datetime.now()
	clip = 0 if delta>=datetime.timedelta(hours=1) else 1
	cost_time = ':'.join(str(delta).split(':')[clip:]).split('.')[0]

	delta *= epochs-epoch-1
	finish = now + delta  
	finish_time=finish.strftime('%Y-%m-%d %H:%M:%S')
	return cost_time,finish_time

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
		optimizer = optim.SGD([x_adv], lr=step_size)

		for i in range(perturb_steps):
			print("Perturb step: ", i)
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