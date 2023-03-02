import cv2
import uuid
import os
from tifffile import TiffFile

#path of tiff file
path = r"X:\Abdallah_NTS_Analysis\554696-1 COUNTED\Experiment.lif - TP-4, 3h Post-Reperfusion, 10X, Z Stack, Area 3-1.tif"

#image path
imgname = r"C:\Users\Abdallah\Desktop\sjdfk"

if not os.path.exists(imgname):
    os.makedirs(imgname)

#Extracts single layers from tiff file
with TiffFile(path) as tif:
    for page in tif.pages:
        img = page.asarray()
        #Converts BGR to RGB format
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Unique image name
        img1name = os.path.join(imgname, "nod" + '.' + '{}.jpg'.format(str(uuid.uuid1())))
        
        #image writer
        cv2.imshow("image", cimg)
        cv2.imwrite(img1name, cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows
