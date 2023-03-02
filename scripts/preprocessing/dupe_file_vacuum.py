import glob
import os
from shutil import copyfile

TrainingFolder = os.path.join(os.getcwd(), 'Tensorflow', 'workspace', 'NFOD', 'images', 'train')
TestFolder = os.path.join(os.getcwd(), 'Tensorflow', 'workspace', 'NFOD', 'images', 'test')

TFileList = list(glob.glob(TrainingFolder+"\*"))

Ext = '.xml'
xmllist = []

for file in TFileList:
    if Ext in file:
        xmllist.append(file)

for dupefile in xmllist:
    for x in TFileList:
        if dupefile in x:
            testdest = dupefile.replace('train', 'test')
            copyfile(dupefile, testdest)
            jpgfile = dupefile.replace(Ext, '.jpg')
            jpgfiledest = jpgfile.replace('train', 'test')
            copyfile(jpgfile, jpgfiledest)
