import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np 
from scipy.stats import truncnorm
import logging
import os
import matplotlib.pyplot as plt
import shutil

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters 
input_size = 784
hidden_size1 = 500
# hidden_size2 = 500
num_classes = 10
num_epochs = 300
batch_size = 200
learning_rate = 0.0005
best_prec1 = 0.0

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


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
																					 train=True, 
																					 transform=transforms.ToTensor(),  
																					 download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
																					train=False, 
																					transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
																					 batch_size=batch_size, 
																					 shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
																					batch_size=batch_size, 
																					shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Fully connected neural network
class NeuralNetrrelu_shallow(nn.Module):
		def __init__(self, input_size, hidden_size, num_classes):
				super(NeuralNetrrelu_shallow, self).__init__()
				self.fc1 = nn.Linear(input_size, hidden_size) 
				self.fc2 = nn.Linear(hidden_size, num_classes)  

				aa = np.random.uniform(0.0,1.0,hidden_size)

				self.a = nn.Parameter(torch.tensor((aa>0.5)*(truncnorm.rvs( (np.tan(35.0/180*np.pi)-1.0)/np.sqrt(3.0), (np.tan(55.0/180*np.pi)-1.0)/np.sqrt(3.0), size=hidden_size)*np.sqrt(3.0)+1.0) \
							+ (aa<=0.5)*(truncnorm.rvs((np.tan(-55.0/180*np.pi)+1.0)/np.sqrt(3.0), (np.tan(-35.0/180*np.pi)+1.0)/np.sqrt(3.0), size=hidden_size)*np.sqrt(3.0)-1.0)).float(),requires_grad=True)
		
		
		def forward(self, x):
				out0 = self.fc1(x)
				a1 = self.a.repeat(out0.shape[0],1)			
				out1 = torch.mul(a1,torch.relu(out0))	
				out = self.fc2(out1)
				return out

# # Fully connected neural network
# class NeuralNet_deep(nn.Module):
# 		def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
# 				super(NeuralNet_deep, self).__init__()
# 				self.fc1 = nn.Linear(input_size, hidden_size1)
# 				self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
# 				self.fc3 = nn.Linear(hidden_size2, num_classes)  

# 				aa = np.random.uniform(0.0,1.0,hidden_size1)
# 				cc = np.random.uniform(0.0,1.0,hidden_size2)

# 				self.a = nn.Parameter(torch.tensor((aa>0.5)*(truncnorm.rvs( (np.tan(35.0/180*np.pi)-1.0)/np.sqrt(3.0), (np.tan(55.0/180*np.pi)-1.0)/np.sqrt(3.0), size=hidden_size1)*np.sqrt(3.0)+1.0) \
# 							+ (aa<=0.5)*(truncnorm.rvs((np.tan(-55.0/180*np.pi)+1.0)/np.sqrt(3.0), (np.tan(-35.0/180*np.pi)+1.0)/np.sqrt(3.0), size=hidden_size1)*np.sqrt(3.0)-1.0)).float(),requires_grad=True)
# 				self.c = nn.Parameter(torch.tensor((cc>0.5)*(truncnorm.rvs( (np.tan(35.0/180*np.pi)-1.0)/np.sqrt(3.0), (np.tan(55.0/180*np.pi)-1.0)/np.sqrt(3.0), size=hidden_size2)*np.sqrt(3.0)+1.0) \
# 							+ (cc<=0.5)*(truncnorm.rvs((np.tan(-55.0/180*np.pi)+1.0)/np.sqrt(3.0), (np.tan(-35.0/180*np.pi)+1.0)/np.sqrt(3.0), size=hidden_size2)*np.sqrt(3.0)-1.0)).float(),requires_grad=True)
		
		
# 		def forward(self, x):
# 				out0 = self.fc1(x)
# 				a1 = self.a.repeat(out0.shape[0],1)			
# 				out1 = torch.mul(a1,torch.relu(out0))	
# 				c1 = self.c.repeat(out1.shape[0],1)			
# 				out2 = torch.mul(c1,torch.relu(out1))	
# 				out = self.fc2(out2)
# 				return out

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
def validate(test_loader, model):	
	with torch.no_grad():
			correct = 0
			total = 0
			for images, labels in test_loader:
					images = images.reshape(-1, 28*28).to(device)
					labels = labels.to(device)
					outputs = model(images)
					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

			# print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
	return 100 * correct / total

if not os.path.exists('./MNISTfiles/'):
		os.makedirs('./MNISTfiles/')
## Uncomment it if you wnt to run ReLU with FCNN	
# model = NeuralNet(input_size, hidden_size1, num_classes).to(device)
# setup_logging(os.path.join('./MNISTfiles/', 'NeuralNetrelu_logger.log'))

# ## Uncomment it if you wnt to run RReLU with FCNN	
model = NeuralNetrrelu_shallow(input_size, hidden_size1, num_classes).to(device)
setup_logging(os.path.join('./MNISTfiles/', 'NeuralNetrrelu_logger.log'))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model (comment this part if you want to only test the model
# If commented, make sure you already have a trained model inside the directory MNISTfiles)
# ==============================================================================
total_step = len(train_loader)
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):  
		# Move tensors to the configured device
		images = images.reshape(-1, 28*28).to(device)
	
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Backprpagation and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		prec1 = accuracy(outputs.data, labels)[0]
	val_prec1 = validate(test_loader, model)
	# remember best prec
	is_best = val_prec1 > best_prec1
	best_prec1 = max(val_prec1, best_prec1)
	if is_best:
		best_epoch = epoch

  ## Change the name of model from 'checkpoint_best_rrelu.th' to 'checkpoint_best_relu.th' for ReLU
	if is_best:
		save_checkpoint({
			'epoch': epoch +1 ,
			'best_epoch': best_epoch,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer': optimizer.state_dict()
		}, is_best, filename=os.path.join('./MNISTfiles/', 'checkpoint_best_rrelu.th'))
		
	logging.info("Epoch- {epoch} \t"
							"Loss- {loss} \t"
							"Prec@1- {prec1} \t"
							"ValPrec@1- {val_prec1}".format(epoch=epoch, loss=loss, prec1=prec1, val_prec1=val_prec1))



# testing the model=====================================================================
## Change the model name appropriately. For ReLU, replace checkpoint_best_rrelu.th with checkpoint_best_relu.th
checkpoint = torch.load("./MNISTfiles/checkpoint_best_rrelu.th")
model.load_state_dict(checkpoint['state_dict'])

zeros = 0.0
slopes = 0.0
zeta = 1.0
for name, p in model.named_parameters():
	if name[-1]=="a":
		max_ = torch.max(torch.abs(p))
		p.data.copy_(torch.tensor([p[i] if torch.abs(p[i])>zeta  else 0.0 for i in range(len(p))])) #0.27 to keep same accuracy
		zeros += p.numel() - p.nonzero().size(0)
		slopes += p.numel()
print("Number of neurons made zero: ", zeros,"Total number of neurons: ", slopes)

val_prec1 = validate(test_loader, model)
print("Validation accuracy: ", val_prec1)

# ## the values of relus
# A = []
# for name, p in model.named_parameters():
# 	if name[-1]=="a" or name[-1]=="c":
# 		print(name)
# 		A = np.concatenate((A,list(p.cpu().detach().numpy())), axis=0)
# print(len(A))
# plt.figure(figsize=(3.5,2.5))
# plt.hist(A, bins=100)
# plt.savefig("hist_FCNN")
# # # exit()
