#!/usr/bin/env python
# coding: utf-8

# In[101]:


import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor


# In[102]:


import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import torchxrayvision as xrv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor
from skimage.io import imread, imsave
import skimage

torch.cuda.empty_cache()
# BORDER_CONSTANT = 0
# BORDER_REFLECT = 2
# crop_size = 224
# scale_size = crop_size * 4

# transforms = albu.Compose([
#   albu.LongestMaxSize(max_size=scale_size),
#   albu.PadIfNeeded(scale_size, scale_size, border_mode=BORDER_CONSTANT),
#   albu.RandomCrop(crop_size, crop_size),
#   albu.OneOf([
#     albu.ShiftScaleRotate( 
#       shift_limit=0.1,
#       scale_limit=0.1,
#       rotate_limit=15,
#       border_mode=BORDER_REFLECT,
#       p=0.5
#     ),
#     albu.Flip(p=0.5),
#     albu.RandomRotate90(p=0.5),     
#   ]),
#   albu.IAAPerspective(scale=(0.02, 0.05), p=0.3),
#   albu.JpegCompression(quality_lower=80),
#   ToTensor()
# ])

# transforms_fn = Augmentor(
#     dict_key="PA",
#     augment_fn=lambda x: transforms(image=x[0][:, :, None])["image"]
# )


# In[103]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer_ImageNet = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# In[104]:


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


# In[105]:


def dataloaders(data_dir, batchsize=4):
    dataset = ImageFolder(data_dir)
    character = [[] for i in range(len(dataset.classes))]
    print(character)
#     print(dataset.samples)
    
    for x, y in dataset.samples:  
        if y == 2:
            character[0].append(x)
        else:
            character[y].append(x)
    print('len data 0',len(character[0]))
    print('len data 1',len(character[1]))
    
    train_inputs = []
    train_labels = []
    
    for i, data in enumerate(character):
        print('data length',len(data))
#         num_sample_train = int(len(data) * ratio[0])
#         num_sample_val = int(len(data) * ratio[1])
#         num_val_index = num_sample_train + num_sample_val

        for x in data[:]:
            train_inputs.append(str(x))
            train_labels.append(i)
               
#     train_inputs, val_inputs, test_inputs = [], [], []
#     train_labels, val_labels, test_labels = [], [], []
#     for i, data in enumerate(character):
#         print('data length',len(data))
#         num_sample_train = int(len(data) * ratio[0])
#         num_sample_val = int(len(data) * ratio[1])
#         num_val_index = num_sample_train + num_sample_val

#         for x in data[:num_sample_train]:
#             train_inputs.append(str(x))
#             train_labels.append(i)
#         for x in data[num_sample_train:num_val_index+1]:
#             val_inputs.append(str(x))
#             val_labels.append(i)
#         for x in data[num_val_index+1:]:
#             test_inputs.append(str(x))
#             test_labels.append(i)
#         print(len(train_inputs))
#         print(len(val_inputs))
#         print(len(test_inputs))
            
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet), batch_size=batchsize, drop_last=False, shuffle=True)
#     val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet), batch_size=batchsize, drop_last=False, shuffle=False)
#     test_dataloader = DataLoader(MyDataset(test_inputs, test_labels, val_transformer_ImageNet), batch_size=batchsize, shuffle=False)
    return train_dataloader
#     return train_dataloader,val_dataloader, test_dataloader


# In[106]:


if __name__ == '__main__':
    data_dir = 'COVID-CT/train_set'
#     train_dataloader,val_dataloader, test_dataloader = fetch_dataloaders(data_dir, [0.7, 0.15, 0.15], batchsize=4)
    train_dataloader = dataloaders(data_dir, batchsize=4)
    val_dataloader = dataloaders('COVID-CT/val_set', batchsize=4)
    test_dataloader = dataloaders('COVID-CT/test_set', batchsize=4)
    


# In[107]:


for batch_index, batch_samples in enumerate(train_dataloader):      
        data, target = batch_samples[0], batch_samples[1]
skimage.io.imshow(data[0,1,:,:].numpy())


# In[108]:


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# alpha = None
device = 'cuda'
def train(optimizer, epoch):
    
    model.train()
    train_loader = train_dataloader
    
    train_loss = 0
    train_correct = 0
    
    for batch_index, batch_samples in enumerate(train_loader):
        
        # move data to device
        data, target = batch_samples[0].to(device), batch_samples[1].to(device)
        data = data[:, 0, :, :]
        data = data[:, None, :, :]
#         data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)
        
        
        optimizer.zero_grad()
        output = model(data)
        
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
#         loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)
        train_loss += criteria(output, target.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
            f = open('COVID-CT/model_result/DenseNet_{}.txt'.format(alpha), 'a+')
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
            f.write('\n')
            f.close()
#             niter = epoch*len(train_loader)+batch_index
#             writer.add_scalar('Train/Loss', loss.data, niter)
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f = open('model_result/DenseNet_alpha_{}.txt'.format(alpha), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.write('\n')
    f.close()


# In[109]:


def test(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    val_loader = val_dataloader
    results = []
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples[0].to(device), batch_samples[1].to(device)
            data = data[:, 0, :, :]
            data = data[:, None, :, :]
#             print(target)
            output = model(data)
            
            test_loss += criteria(output, target.long())
            pred = output.argmax(dim=1, keepdim=True)
#             print('target',target.long()[:, 2].view_as(pred))
            correct += pred.eq(target.long().view_as(pred)).sum().item()
#             TP += ((pred == 1) & (target.long()[:, 2].view_as(pred).data == 1)).cpu().sum()
#             TN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
# #             # FN    predict 0 label 1
#             FN += ((pred == 0) & (target.long()[:, 2].view_as(pred) == 1)).cpu().sum()
# #             # FP    predict 1 label 0
#             FP += ((pred == 1) & (target.long()[:, 2].view_as(pred) == 0)).cpu().sum()
#             print(TP,TN,FN,FP)
            
            
#             print(output[:,1].cpu().numpy())
#             print((output[:,1]+output[:,0]).cpu().numpy())
#             predcpu=(output[:,1].cpu().numpy())/((output[:,1]+output[:,0]).cpu().numpy())
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            targetlist=np.append(targetlist,targetcpu)
           
          
#         print('test epoch is',epoch)
# #         print('pred',predlist)
# #         print('target',targetlist)
#         print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
#         print('TP+FP',TP+FP)
#         p = TP/ (TP + FP)
#         print('precision',p)
#         p = TP / (TP + FP)
#         r = TP / (TP + FN)
#         print('recall',r)
#         F1 = 2 * r * p / (r + p)
#         acc = (TP + TN) / (TP + TN + FP + FN)
#         print('F1',F1)
#         print('acc',acc)
#         AUC = roc_auc_score(targetlist, predlist)
#         print('AUC', AUC)
##        AUC estimate
       
    val_loss /= len(test_loader.dataset)
    

    # Display results
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))
    
#     r=1;p=1; acc=1; AUC=1;
    return targetlist, predlist
    
    # Write to tensorboard
#     writer.add_scalar('Test Accuracy', 100.0 * correct / len(test_loader.dataset), epoch)


# In[ ]:


# %CheXNet pretrain
# class DenseNet121(nn.Module):
#     """Model modified.

#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.

#     """
#     def __init__(self, out_size):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=True)
#         num_ftrs = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.densenet121(x)
#         return x
  

# device = 'cuda'
# CKPT_PATH = 'model.pth.tar'
# N_CLASSES = 14

# DenseNet121 = DenseNet121(N_CLASSES).cuda()

# CKPT_PATH = './CheXNet/model.pth.tar'

# if os.path.isfile(CKPT_PATH):
#     checkpoint = torch.load(CKPT_PATH)        
#     state_dict = checkpoint['state_dict']
#     remove_data_parallel = False


#     pattern = re.compile(
#         r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
#     for key in list(state_dict.keys()):
#         match = pattern.match(key)
#         new_key = match.group(1) + match.group(2) if match else key
#         new_key = new_key[7:] if remove_data_parallel else new_key
#         new_key = new_key[7:]
#         state_dict[new_key] = state_dict[key]
#         del state_dict[key]


#     DenseNet121.load_state_dict(checkpoint['state_dict'])
#     print("=> loaded checkpoint")
# #     print(densenet121)
# else:
#     print("=> no checkpoint found")

# # for parma in DenseNet121.parameters():
# #         parma.requires_grad = False
# DenseNet121.densenet121.classifier._modules['0'] = nn.Linear(in_features=1024, out_features=2, bias=True)
# DenseNet121.densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# # print(DenseNet121)
# model = DenseNet121.to(device)


# In[110]:


class DenseNetModel(nn.Module):

    def __init__(self):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits
    
model = DenseNetModel().to(device)
# print(model)


# In[ ]:


# valiation
# 1dense
bs = 4
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
val_loader = val_dataloader
vote_pred = np.zeros((1,len(val_dataloader.dataset)))

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 3000
for epoch in range(1, total_epoch+1):
    train(optimizer, epoch)
    
    targetlist, predlist = test(epoch)
    vote_pred = vote_pred + predlist 
    
    p_list.append(p)
    r_list.append(r)
    acc_list.append(acc)
    AUC_list.append(AUC)

    # Save model
    if epoch == total_epoch:
       torch.save(model.state_dict(), "COVID-CT/model_backup/new_split_DenseNet_{}.pt".format(epoch))  

    if epoch % 10 == 0:
        r_ave = sum(r_list)/len(r_list)
        p_ave = sum(p_list)/len(p_list)
        acc_ave = sum(acc_list)/len(acc_list)
        AUC_ave = sum(AUC_list)/len(AUC_list)
        F1_ave = 2 * r_ave * p_ave / (r_ave + p_ave)
        r_list = []
        p_list = []
        acc_list = []
        AUC_list = []
        
        # major vote
        vote_pred[vote_pred <= 5] = 0
        vote_pred[vote_pred > 5] = 1
        
        print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        
        
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, predlist)
        print('AUC', AUC)
        
        
        f = open('COVID-CT/model_result/DenseNet_.txt'.format(alpha), 'a+')
        f.write('precision, recall, F1, acc= \n')
        f.writelines(str(p))
        f.writelines('\n')
        f.writelines(str(r))
        f.writelines('\n')
        f.writelines(str(F1))
        f.writelines('\n')
        f.writelines(str(acc))
        f.writelines('\n')
        f.close()
        
        
        vote_pred = np.zeros((1,len(val_loader.dataset)))
        print('vote_pred',vote_pred)
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open('model_result/new_split_DenseNet_alpha_{}.txt'.format(alpha), 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()
# state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
#                  'optimizer': optimizer.state_dict(), 'scheduler' : scheduler}
# torch.save(state, "model_backup/AlexNetAdamState")


# In[ ]:


# test
bs = 4
import warnings
warnings.filterwarnings('ignore')

r_list = []
p_list = []
acc_list = []
AUC_list = []
# TP = 0
# TN = 0
# FN = 0
# FP = 0
test_loader = test_dataloader
vote_pred = np.zeros((1,len(test_loader.dataset)))

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
scheduler = StepLR(optimizer, step_size=1)

total_epoch = 20
for epoch in range(1, total_epoch+1):

    targetlist, predlist = test(epoch)
    vote_pred = vote_pred + predlist 
    
    p_list.append(p)
    r_list.append(r)
    acc_list.append(acc)
    AUC_list.append(AUC)

    # Save model
    if epoch == total_epoch:
       torch.save(model.state_dict(), "COVID-CT/model_backup/new_split_DenseNet_{}.pt".format(epoch))  

    if epoch % 10 == 0:
        r_ave = sum(r_list)/len(r_list)
        p_ave = sum(p_list)/len(p_list)
        acc_ave = sum(acc_list)/len(acc_list)
        AUC_ave = sum(AUC_list)/len(AUC_list)
        F1_ave = 2 * r_ave * p_ave / (r_ave + p_ave)
        r_list = []
        p_list = []
        acc_list = []
        AUC_list = []
        
        # major vote
        vote_pred[vote_pred <= 5] = 0
        vote_pred[vote_pred > 5] = 1
        
        print('vote_pred', vote_pred)
        print('targetlist', targetlist)
        TP = ((vote_pred == 1) & (targetlist == 1)).sum()
        TN = ((vote_pred == 0) & (targetlist == 0)).sum()
        FN = ((vote_pred == 0) & (targetlist == 1)).sum()
        FP = ((vote_pred == 1) & (targetlist == 0)).sum()
        
        
        print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
        print('TP+FP',TP+FP)
        p = TP / (TP + FP)
        print('precision',p)
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        print('recall',r)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('F1',F1)
        print('acc',acc)
        AUC = roc_auc_score(targetlist, predlist)
        print('AUC', AUC)
        
        
        f = open('COVID-CT/model_result/DenseNet_.txt'.format(alpha), 'a+')
        f.write('precision, recall, F1, acc= \n')
        f.writelines(str(p))
        f.writelines('\n')
        f.writelines(str(r))
        f.writelines('\n')
        f.writelines(str(F1))
        f.writelines('\n')
        f.writelines(str(acc))
        f.writelines('\n')
        f.close()
        
        
        vote_pred = np.zeros((1,len(test_loader.dataset)))
        print('vote_pred',vote_pred)
        print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))

        f = open('model_result/new_split_DenseNet_alpha_{}.txt'.format(alpha), 'a+')
        f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}'.format(
        epoch, r, p, F1, acc, AUC))
        f.close()
# state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
#                  'optimizer': optimizer.state_dict(), 'scheduler' : scheduler}
# torch.save(state, "model_backup/AlexNetAdamState")


# In[ ]:




