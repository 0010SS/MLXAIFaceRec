# build the recognizer based on the trained model

import cv2, time
import serial

# initialize arduino
#ArduinoSerial = serial.Serial('/dev/cu.usbserial-11 0', 9600, timeout=0.1)
time.sleep(1)

# initialization
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
cascadePath = 'OpenCVTrainedPara.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX

# initiate ID counter
id = 0

# names related to the IDs
names = ['Hardy', 'David', 'Humphrey', 'William', 'Unknowns', 'Rico', 'Harry', 'Ben']

# initialize the camera
cam = cv2.VideoCapture(0)
cam.set(3, 1440)
cam.set(4, 680)

# define min size to be considered as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

# print out a message
print("start running the recognizer")

while cam.isOpened():
    print(cv2.getBuildInformation())
    ret, img = cam.read()
    print(ret)
    print(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x,y,w,h) in faces:

        # sending coordinates to Arduino
        string = 'X{0:d}Y{1:d}'.format((x + w // 2), (y + h // 2))
        print(string)
        #ArduinoSerial.write(string.encode('utf-8'))

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w]) # predict whether the face is known
        confidence = round(100 - confidence)

        # if confidence is less than
        if confidence > 35:
            id = names[id-1]
            confidence = "  {0}%".format(confidence)
        else:
            id = "unknown"
            confidence = "  {0}%".format(confidence)

        # display the name and confidence level
        cv2.putText(
            img,
            str(id),
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x+5, y+h-5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()