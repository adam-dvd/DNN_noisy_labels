"""Script adapted from
Code for ICCV2019 Paper ["O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks"](https://ieeexplore.ieee.org/document/9008796)
@INPROCEEDINGS{huang2019o2unet,
  author={Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang},
  booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)}, 
  title={O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks}, 
  year={2019},
  pages={3325-3333},
  doi={10.1109/ICCV.2019.00342},
  }
"""

import os
import torch 
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from data.mask_data import Mask_Select

from cnn import CNN
from active_passive_loss import NCEandRCE
from data.cifar import CIFAR4

class Arguments:
	def __init__(
			self, 
			result_dir: str = '~/', 
			noise_rate: float = 0.2, 
			forget_rate: float = None,
			dataset: str = 'cifar10',
			n_epoch: int = 100,
			seed: int = 2,
			batch_size: int = 128,
			network: str = 'my_cnn',
			transforms: str = 'false',
			unstabitily_batch: int = 16
	):
		self.result_dir = result_dir
		self.noise_rate = noise_rate
		self.forget_rate = forget_rate
		self.dataset = dataset
		self.n_epoch = n_epoch
		self.seed = seed
		self.batch_size = batch_size
		self.network = network
		self.transforms = transforms
		self.unstabitily_batch = unstabitily_batch

def adjust_learning_rate(optimizer, epoch,max_epoch=200):
	if epoch < 0.25 * max_epoch:
		lr = 0.01
	elif epoch < 0.5 * max_epoch:
		lr = 0.005
	else:
		lr = 0.001
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def evaluate(test_loader, model1):
	model1.eval()
	correct1 = 0
	total1 = 0
	for images, labels, _ in test_loader:
		images = Variable(images).cuda()
		#print images.shape
		logits1 = model1(images)
		outputs1 = F.log_softmax(logits1, dim=1)
		_, pred1 = torch.max(outputs1.data, 1)
		total1 += labels.size(0)
		correct1 += (pred1.cpu() == labels).sum()
	acc1 = 100 * float(correct1) / float(total1)
	model1.train()

	return acc1

def first_stage(network,test_loader,train_dataset, args, noise_or_not, active_passive_loss=False, filter_mask=None):
	if filter_mask is not None:#third stage
		train_loader_init = torch.utils.data.DataLoader(dataset=Mask_Select(train_dataset,filter_mask),
													batch_size=128,
													num_workers=2,
													shuffle=True,pin_memory=True)
	else:
		train_loader_init = torch.utils.data.DataLoader(dataset=train_dataset,
														batch_size=128,
														num_workers=2,
														shuffle=True, pin_memory=True)
	save_checkpoint=args.network+'_'+args.dataset+'_'+str(args.noise_rate)+'.pt'
	if  filter_mask is not None and os.path.isfile(save_checkpoint):
		print ("restore model from %s.pt"%save_checkpoint)
		network.load_state_dict(torch.load(save_checkpoint))
	ndata=train_dataset.__len__()
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	if active_passive_loss:
		criterion = NCEandRCE(alpha=1,beta=1,num_classes=4).cuda()
	else:
		criterion = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	accuracies = []
	loss_l = []
	for epoch in range(1, args.n_epoch):
		# train models
		globals_loss = 0
		network.train()
		with torch.no_grad():
			accuracy = evaluate(test_loader, network)
		#example_loss = np.zeros_like(noise_or_not, dtype=float)
		lr=adjust_learning_rate(optimizer1,epoch,args.n_epoch)
		for i, (images, labels, indexes) in enumerate(train_loader_init):
			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 = criterion(logits, labels)

			#for pi, cl in zip(indexes, loss_1):
			#	example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()
			loss_1 = loss_1.mean()

			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()
		loss_l.append(globals_loss /ndata)
		accuracies.append(accuracy)
		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss /ndata, "test_accuracy:%f" % accuracy)
		if filter_mask is None:
			torch.save(network.state_dict(), save_checkpoint)
	return loss_l, accuracies


def second_stage(network,test_loader, train_dataset, args, noise_or_not, max_epoch=250):
	train_loader_detection = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=16,
											   num_workers=2,
											   shuffle=True)
	optimizer1 = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	criterion=torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-1).cuda()
	moving_loss_dic=np.zeros_like(noise_or_not)
	ndata = train_dataset.__len__()

	noise_accuracies = []
	mask_l = []
	lr_l = []
	loss_1_sorted_l = []
	for epoch in range(1, max_epoch):
		# train models
		globals_loss=0
		network.train()
		with torch.no_grad():
			accuracy=evaluate(test_loader, network)
		example_loss= np.zeros_like(noise_or_not,dtype=float)

		t = (epoch % 10 + 1) / float(10)
		lr = (1 - t) * 0.01 + t * 0.001

		for param_group in optimizer1.param_groups:
			param_group['lr'] = lr

		for i, (images, labels, indexes) in enumerate(train_loader_detection):

			images = Variable(images).cuda()
			labels = Variable(labels).cuda()

			logits = network(images)
			loss_1 =criterion(logits,labels)

			for pi, cl in zip(indexes, loss_1):
				example_loss[pi] = cl.cpu().data.item()

			globals_loss += loss_1.sum().cpu().data.item()

			loss_1 = loss_1.mean()
			optimizer1.zero_grad()
			loss_1.backward()
			optimizer1.step()
		loss_by_sample = example_loss
		example_loss=example_loss - example_loss.mean()
		moving_loss_dic=moving_loss_dic+example_loss

		ind_1_sorted = np.argsort(moving_loss_dic)
		loss_1_sorted = moving_loss_dic[ind_1_sorted]

		if args.forget_rate is None:
			forget_rate=args.noise_rate
		else:
			forget_rate=args.forget_rate
		remember_rate = 1 - forget_rate
		num_remember = int(remember_rate * len(loss_1_sorted))

		noise_accuracy=np.sum(noise_or_not[ind_1_sorted[num_remember:]]) / float(len(loss_1_sorted)-num_remember)
		mask = np.ones_like(noise_or_not,dtype=np.float32)
		mask[ind_1_sorted[num_remember:]]=0

		top_accuracy_rm=int(0.9 * len(loss_1_sorted))
		top_accuracy= 1-np.sum(noise_or_not[ind_1_sorted[top_accuracy_rm:]]) / float(len(loss_1_sorted) - top_accuracy_rm)

		noise_accuracies.append(noise_accuracy)
		mask_l.append(mask)
		loss_1_sorted_l.append(loss_by_sample[ind_1_sorted])
		lr_l.append(lr)
		print ("epoch:%d" % epoch, "lr:%f" % lr, "train_loss:", globals_loss / ndata, "test_accuarcy:%f" % accuracy,"noise_accuracy:%f"%(1-noise_accuracy),"top 0.1 noise accuracy:%f"%top_accuracy)



	return mask, lr_l, mask_l, noise_accuracies, loss_1_sorted_l
