# the trainer for face recognition

import cv2
import numpy as np
from PIL import Image
import os

# function to obtain the images
def getData(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] # store the path of each image into a list
    faceSamples = [] # store the vector of each face
    IDs = [] # store the id of each face
    for imagePath in imagePaths: # considering each image sample
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img, 'uint8') # convert the image into an np.array
        ID = int(os.path.split(imagePath)[-1].split(".")[1]) # obtain the ID
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            IDs.append(ID)
    return faceSamples, IDs

# path for dataset
path = "ImageDataset"

# create the recognizer and the detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("OpenCVTrainedPara.xml")

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getData(path)
recognizer.train(faces, np.array(ids)) # training

#save the model into trainer.yml
recognizer.write('trainer.yml')

#print the number of faces trained
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))