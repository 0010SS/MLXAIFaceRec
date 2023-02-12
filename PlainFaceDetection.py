import cv2

faceCascade = cv2.CascadeClassifier("Model/OpenCVTrainedPara.xml")

cap = cv2.VideoCapture(0) # open the camera
cap.set(3, 640)  # set Width
cap.set(4, 480)  # set Height

while True:
    success, img = cap.read() # read in the information of the camera now
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(  # classifier function
        gray,  # the input gray scale image
        scaleFactor=1.2,  # specify how much the image size is reduced at each image scale
        minNeighbors=5,  # specify how many neighbors each candidate rectangle should have
        minSize=(100, 100)  # specify the minimum rectangle size to be detected as a face
    )

    for (x, y, w, h) in faces:  # for every face detected
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw a rectangle

        cv2.imshow('video', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

# cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()