import cv2
import numpy as np
import os
from os import walk
from glob import glob
from os import listdir
from os.path import isfile, join
from alive_progress import alive_bar

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:

            allFiles.append(fullPath)
                
    return allFiles  

def get_square(image,square_size):
    height = np.size(image, 0)
    width  = np.size(image, 1)
    if height == square_size and width == square_size:
        return image
    else:
        if(height>width):
            differ=height
        else:
            differ=width
        differ+=4
        mask = np.zeros((differ,differ, 3), dtype="uint8")   
        x_pos=int((differ-width)/2)
        y_pos=int((differ-height)/2)
        mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
        mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_CUBIC)
        return mask 

dirName     = './dataset'
listOfFiles = getListOfFiles(dirName)
with alive_bar(len(listOfFiles), force_tty=True) as bar:
    for img in listOfFiles:
        imgcv = cv2.imread(img)
        if imgcv is not None:
            imgcv = get_square(imgcv,224)
            try:
                cv2.imwrite(img,imgcv)
            except:
                print("Cannot write image:")
                print(img)
        else:
            print("Cannot read image:")
            print(img)
        bar()
