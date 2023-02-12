# this file reads in the image data and stores it in ImageDataset file

import cv2

faceCascade = cv2.CascadeClassifier('OpenCVTrainedPara.xml') # load the trained model with parameters

cap = cv2.VideoCapture(0) # open the camera
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# initialize the face count
count = 1

while True:
    ret, img = cap.read() # read in the information of the camera now
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # set to gray scale

    faces = faceCascade.detectMultiScale( # classifier function
        gray, # the input gray scale image
        scaleFactor=1.2, # specify how much the image size is reduced at each image scale
        minNeighbors=5, # specify how many neighbors each candidate rectangle should have
        minSize=(100, 100) # specify the minimum rectangle size to be detected as a face
    )

    for (x, y, w, h) in faces: # for every face detected
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) # draw a rectangle
        count += 1 # number of face sampling +1

        # save the captured image
        cv2.imwrite("ImageDataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('video', img)

        if count == 200:  # controls number of training examples
            break

        count += 1

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()