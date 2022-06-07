import cv2

# load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # read the frame
    _, img = cap.read()

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect the face
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # draw rectangle around the face
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # display
    cv2.imshow('Face Detection', img)

    # stop if escape key (Esc) is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()