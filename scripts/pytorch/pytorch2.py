import os
import torch
import cv2
import glob
import torchvision.transforms as transforms
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import torchvision
import torch
import pickle
import random
import time
import torch.optim as optim
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

#Sorting the jpg list and xml list
def img_path(path):
    imglist = []
    for jpg_file in glob.glob(path + '/*.jpg'):
        imglist.append(jpg_file)
    return imglist

#input xml annotation folder path and converts to df
def xml_to_list(path):
    xml_dict = {}
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        annotations = []
        for member in root.findall('object'):
            annotation = [
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text)
            ]
            annotations.append(annotation)
        if filename in xml_dict:
            xml_dict[filename].extend(annotations)
        else:
            xml_dict[filename] = annotations

    xml_list = [[filename, annotations] for filename, annotations in xml_dict.items()]
    
    return xml_list



#one-hot encoding dictionary for the neutrophil class
class_dict = {
    'neutrophil' : 0
    }

#obtaining annotated data then one-hot encoding class
master_folder = r"C:\Users\WinshipLab\Desktop\Abdallah\Oct11tfod\Tensorflow\workspace\NFOD\images"
train_fol = os.path.join(master_folder, 'train')
val_fol = os.path.join(master_folder, 'test')

#Training annotation dataset dataframe
tlist = xml_to_list(train_fol)
#tdf['class'] = tdf['class'].apply(lambda x: class_dict[x])
#tdf = tdf[['xmin', 'ymin', 'xmax', 'ymax', 'class']]

print(tlist)
#Validation annotation dataset dataframe
vlist = xml_to_list(val_fol)
#vdf['class'] = vdf['class'].apply(lambda x: class_dict[x])
#vdf = vdf[['xmin', 'ymin', 'xmax', 'ymax', 'class']]

#this function produces 3 lists, the class labels, the boxes normalized, the img values normalized
#only accepts df with structure column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
def preprocess(xmllist, dataset_type='train'):
    
    dataset_types = 'train', 'test'
    if dataset_type not in dataset_types:
        raise ValueError("Invalid dataset type. Expected one of: %s" % dataset_types)
    else:
        imgpath = os.path.join(master_folder, dataset_type)
    
    labels = [0]
    boxes = []
    imglist = []
    total = 0
    
    for annotation in xmllist:
        #counts the number of annotations starting at index 1 (index 0 is imgname)
        total += len(annotation[1])
        for elem in annotation: 
            #checks to see if elem is a list
            if isinstance(elem, list):
                #Normalize bounding box cordinates in range [0, 1] by dividing each cordinate by 1024 (that is the size of the image)
                new_elem = [[float(xmin) / 1024,
                            float(ymin) / 1024,
                            float(xmax) / 1024, 
                            float(ymax) / 1024] 
                            for xmin, ymin, xmax, ymax in elem]
                boxes.append(new_elem)
            else:
                image = cv2.imread(os.path.join(imgpath, elem))
                
                #convert image from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                #normalize
                image = image.astype('float') / 255
                
                imglist.append(image)
    
    labels = labels * total
    
    return labels, boxes, imglist        

#load data to check if its working properly
labels, boxes, imglist = preprocess(tlist, dataset_type='train')

vallabels, valboxes, valimglist = preprocess(vlist, dataset_type='test')

trainlabels = np.array(labels) 
trainboxes = np.array(boxes)
trainimglist = np.array(imglist)
#vallabels, valboxes, valimglist = np.array(vallabels, dtype=float), np.array(valboxes, dtype=float), np.array(valimglist, dtype=float)

# #shuffle data and then unzip
# combined_list = list(zip(imglist, boxes, labels))
# random.shuffle(combined_list)
# imglist, boxes, labels = zip(*combined_list)

# #creating matplotlib figure
# plt.figure(figsize=(64, 64))

# #Generate random sample of images
# random_range  = random.sample(range(1, len(imglist)), 9)
# img_size = 1024

# for itr, i in enumerate(random_range, 1):
    
#     nimg = imglist[i]
    
#     for box in boxes[i]:
#         a1, b1, a2, b2 = box
        
#         #rescaling the bb to match imgsize
#         x1 = a1 * img_size
#         x2 = a2 * img_size
#         y1 = b1 * img_size
#         y2 = b2 * img_size

#         #draw the bounding boxes on the image
#         cv2.rectangle(nimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#     img = np.clip(imglist[i], 0, 1)
#     plt.subplot(3, 3, itr)
#     plt.imshow(img)
#     plt.axis('off')
    
# plt.show()
        
#Creating the dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, trainimglist, trainlabels, trainboxes):
        #Permutes BS, C, H, W
        self.images = torch.permute(torch.from_numpy(trainimglist), (0,3,1,2)).float()
        self.labels = torch.from_numpy(trainlabels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(trainboxes).float()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.images[idx],
                self.labels[idx],
                self.boxes[idx])

class ValDataset(Dataset):
    def __init__(self, valimglist, vallabels, valboxes):
        
        #self.images = torch.permute(torch.from_numpy(valimglist), (0,3,1,2)).float()
        self.images = torch.from_numpy(valimglist).float().permute(0,3,1,2)
        self.labels = torch.from_numpy(vallabels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(valboxes).float()
    
#specifying the data augs for dataset
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomCrop(size=(512,512)),
#     transforms.RandomRotation(degrees=(0, 180)),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomAdjustSharpness(sharpness_factor=2),
#     transforms.ColorJitter(brightness=0.5, saturation=0.3),
#     transforms.ToTensor()
# ])
    
#Select GPU for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        #Convolutional Layers and pooling layers for 2P images, input (batchsize(4), num_channels (3), height(1024), width(1024))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding_mode='reflect')
        self.batch_norm1 = nn.BatchNorm2d(num_features=8, momentum=0.9)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #conv2 (4, 3, 512, 512) halved becaused pooling layer (kern 2, stri 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding_mode='reflect')
        self.batch_norm2 = nn.BatchNorm2d(num_features=16, momentum=0.9)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #conv3 (4, 3, 256, 256) havled because pooling layer (kern 2, stri 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding_mode='reflect')
        self.batch_norm3 = nn.BatchNorm2d(num_features=32, momentum=0.9)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #conv4 (4, 3, 128, 128) halved because pooling layer (kern2, stri 2) 
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding_mode='reflect')
        self.batch_norm4 = nn.BatchNorm2d(num_features=64, momentum=0.9)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), padding_mode='reflect')
        self.batch_norm5 = nn.BatchNorm2d(num_features=256, momentum=0.9)
        self.pool5 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.dropout5 = nn.Dropout(0.2)
        
        #Connecting CL to dense layers (DL) for bounding box prediction
        self.bfc1 = nn.Linear(in_features=(256*16*16), out_features=2500)
        self.bfc2 = nn.Linear(in_features=2500, out_features=500)
        self.bfc3 = nn.Linear(in_features=500, out_features=50)
        self.bout = nn.Linear(in_features=50, out_features=4)
        
        # Class prediction
        self.cfc1 = nn.Linear(in_features=(256*16*16), out_features=2500)
        self.cfc2 = nn.Linear(in_features=2500, out_features=50)
        self.cout = nn.Linear(in_features=50, out_features=2)
        
    def forward(self, i):
        i = self.conv1(i)
        i = F.relu(i)
        i = self.batch_norm1(i)
        i = self.pool1(i)
        
        i = self.conv2(i)
        i = F.relu(i)
        i = self.batch_norm2(i)
        i = self.pool2(i)
        
        i = self.conv3(i)
        i = F.relu(i)
        i = self.batch_norm3(i)
        i = self.pool3(i)
        
        i = self.conv4(i)
        i = F.relu(i)
        i = self.batch_norm4(i)
        i = self.pool4(i)
        i = self.dropout4(i)
        
        i = self.conv5(i)
        i = F.relu(i)
        i = self.batch_norm5(i)
        i = self.pool5(i)
        i = self.dropout5(i)
        
        i = torch.flatten(i, start_dim=1)
        
        #fully connected layers after the convolutions
        class_i = self.cfc1(i)
        class_i = F.relu(class_i)
        
        class_i = self.cfc2(class_i)
        class_i = F.relu(class_i)
        
        class_i = F.softmax(self.cout(class_i), dim=1)
        
        box_i = self.bfc1(i)
        box_i = F.relu(box_i)

        box_i = self.bfc2(box_i)
        box_i = F.relu(box_i)

        box_i = self.bfc3(box_i)
        box_i = F.relu(box_i)
        
        box_i = self.bout(box_i)
        box_i = F.sigmoid(box_i)

        return [class_i, box_i]
    
    
#Instantiate the network and send model to GPU
model = Network()
model = model.to(device)

dataset = Dataset(trainimglist, trainlabels, trainboxes)
valdataset = ValDataset(valimglist, vallabels, valboxes)

#function for determining the number of correct predictions for each batch of training
def get_num_correct(preds, labels):
    return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

valdataloader = DataLoader(valdataset, batch_size=8, shuffle=True)

#model training function
def train(model):
    #define optim
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    num_of_epochs = 50
    epochs = []
    losses = []
    start = time.time()
    for epoch in range(num_of_epochs):
        cnt = 0
        tot_loss = 0
        tot_correct = 0
        train_start = time.time()
        
        #train mode
        model.train()
        for batch, (x, y, z) in enumerate(dataloader):
            #if this doesnt work try:
            #for anno in y:
                #anno.to(device)
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            [y_pred, z_pred] = model(x)
            
            #computing loss
            class_loss = F.cross_entropy(y_pred, y)
            box_loss = F.mse_loss(z_pred, z)
            (box_loss + class_loss).backward()
            
            optimizer.step()
            print("Train batch:", batch+1, "epoch:", epoch, " ",(time.time()-train_start) / 60, end= '\r')
            
        model.eval()
        for batch, (x, y, z) in enumerate(valdataloader):
            x, y, z = x.to(device), y.to(device), z.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                [y_pred, z_pred] = model(x)
                
                #computing loss
                class_loss = F.cross_entropy(y_pred, y)
                box_loss = F.mse_loss(z_pred, z)
                
            tot_loss += (class_loss.item() + box_loss.item())
            tot_correct += get_num_correct(y_pred, y)
            print("Train batch:", batch+1, "epoch:", epoch, " ",(time.time()-train_start) / 60, end= '\r')
            torch.save(model.state_dict(), "models/model_ep"+str(epoch+1)+".pth")
            
train(model)