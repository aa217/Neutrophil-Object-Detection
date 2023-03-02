import glob
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np


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

fol_path = "D:/Abdallah/Venvs/tvenv/notebook/imgs"

xmllist = (xml_to_list(fol_path))

for annot in xmllist:
    for bb in annot:
        if isinstance(bb, list):
            if len(bb) == 0:
                print("empty list found")
            else:
                print('None')    

imglist = []
boxes = []
labels= []
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
            image = cv2.imread(os.path.join(fol_path, elem))
            
            #convert image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #normalize
            image = image.astype('float') / 255
            
            imglist.append(image)

# for box in boxes:
#     print(len(box))

# random_range  = random.sample(range(1, len(imglist)), 9)
# img_size = 1024
# for itr, i in enumerate(random_range, 1):
    
#     nimg = imglist[i]
    
#     for box in nboxes[i]:
#         a1, b1, a2, b2 = box
        
#         #rescaling the bb to match imgsize
#         x1 = a1 * img_size
#         x2 = a2 * img_size
#         y1 = b1 * img_size
#         y2 = b2 * img_size

#         #draw the bounding boxes on the image
#         cv2.rectangle(nimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
#     img = np.clip(imglist[i], 0, 1)
#     plt.subplot(3, 3, itr)
#     plt.imshow(img)
#     plt.axis('off')

# plt.show()

    
