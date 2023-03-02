import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomCrop, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomAdjustSharpness, PILToTensor, ConvertImageDtype, ColorJitter, Compose
from torchvision.io import read_image
import glob
import xml.etree.ElementTree as ET
from engine import train_one_epoch, evaluate
import utils

fol_path = "C:/Users/WinshipLab/Desktop/Abdallah/Oct11tfod/Tensorflow/workspace/NFOD/images/train"

#input xml annotation folder path and converts to df
def xml_to_df(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

#Sorting the jpg list and xml list
def img_path(path):
    imglist = []
    for jpg_file in glob.glob(path + '/*.jpg'):
        imglist.append(jpg_file)
    return imglist
    
#Label encoding dictionary for the neutrophil class
class_dict = {
    'neutrophil' : 0
    }

class NeutrophilDataset(Dataset):
    def __init__(self, img_paths, annotations, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.annotations = annotations
    
    def __getitem__(self, index):
        #Calling xml_to_df
        df_train = xml_to_df(fol_path)

        #changes class in xml_df to label encoding
        df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

        #select xmin ymin xmax ymax
        bb_cords = df_train[['xmin', 'ymin', 'xmax', 'ymax']]

        #convert this df to a numpyarray
        bb_cords_np = bb_cords.values

        #convert df to torch.tensor
        boxes = torch.as_tensor(bb_cords_np, dtype=torch.float32)

        #grab the bb_num and convert to torch.ones
        num_bb = len(df_train[['filename']])
        labels = torch.ones((num_bb,), dtype=torch.int64)
        
        #sorting data
        target = {}
        target["boxes"] =  boxes
        target["labels"] = labels
        
        if self.transform is not None:
            img, target = self.transform(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)
    
    #applying different data augs
    def get_transform(train):
        transforms = []
        transforms.append(PILToTensor())
        transforms.append(ConvertImageDtype(torch.float))
        if train:
            transforms.append(RandomHorizontalFlip(p=0.5))
            transforms.append(RandomCrop(size=(512,512)))
            transforms.append(RandomRotation(degrees=(0, 180)))
            transforms.append(RandomVerticalFlip(p=0.5))
            transforms.append(RandomAdjustSharpness(sharpness_factor=2))
            transforms.append(ColorJitter(brightness=0.5, saturation=0.3))
        return Compose(transforms)
    

#main function that performs training and validation
def main():
        
    #Train on the GPU or the CPU depending on which is present
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    #Number of classes dataset has
    num_classes = 1
        
    #call our dataset
    dataset = 