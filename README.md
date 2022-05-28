                                           **FACE RECOGNITION BASED ATTENDANCE SYSTEM**
                                                  
Python implementation of simple face recognition based attendance system using face_recognition library.

A real time face recognition of students for their attendance. The attendance record is stored in an excel sheet.

**Basic project flow**

local database of face and their encodings ------> face recognition--------> mark attendance in excel sheet

**Methodology of the system:** 

•Capture a video and check each frame for person. • If any person is detected , detect the face and crop the frame around his face. • Generate facial features of that face and match these with the local database. • If the facial features are matched get the name of the person from the local database. • Get the date and name of the person detected and update the attendance in the excel sheet.

**Methodology for face recognition:**

• Capture a video and process each other frame. • Resize the image to 1/4th of the original frame. • Convert the image from BGR to RGB. • generate a 128 byte array of data for each face detected. • Compare this array with the existing arrays in the local database. • Calculate Euclidian distance from each face in the local database and get the index of minimum distance. • Get the name of the best match index.
