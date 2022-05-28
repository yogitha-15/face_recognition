import cv2
import numpy as np
import face_recognition
import os
import streamlit as st

st.title("Face Recognition System")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
path = 'IMAGES'
IMAGES = []
personNames = []
myList = os.listdir(path)

for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    IMAGES.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])

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

encodeListKnown = faceEncodings(IMAGES)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while run:
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

    FRAME_WINDOW.image(frame)

else:
    st.write('Stopped')