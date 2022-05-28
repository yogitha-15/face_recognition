import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'IMAGES'

#Path not existing might give out an error
if not os.path.exists(path):
    os.mkdir(path)

IMAGES = []
personNames = []
myList = os.listdir(path)


for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    IMAGES.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])

#Note images and personNames can remain empty
print("List of people onboard")
print(personNames)


def faceEncodings(IMAGES):
    encodeList = []
    for img in IMAGES:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
        except IndexError as e:
            print("Error Code: ", e, " No face found in the image for encoding")
        else:
            encodeList.append(encode)
    return encodeList


def attendance(name):
    try:
        if not os.path.exists('Attendance.csv'):
            with open('Attendance.csv','w') as f:
                pass
        else:
            with open('Attendance.csv', 'r+') as f:
                myDataList = f.readlines()
                nameList = []
                for line in myDataList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                if name not in nameList:
                    time_now = datetime.now()
                    tStr = time_now.strftime('%H:%M:%S')
                    dStr = time_now.strftime('%d/%m/%Y')
                    f.writelines(f'\n{name},{tStr},{dStr}')
    except FileNotFoundError as e:
        print("Error Code: ", e, " File not found and also couldn't be automatically created, Please create attendance sheet :)")


encodeListKnown = faceEncodings(IMAGES)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()