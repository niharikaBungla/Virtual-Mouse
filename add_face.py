import face_recognition
import cv2
import numpy as np

def resize(img ,size):
    width = int(img.shape[1]*size)
    height= int(img.shape[0] *size)
    dimension= (width ,height)
    return cv2.resize(img ,dimension ,interpolation=cv2.INTER_AREA)

imgElon = face_recognition.load_image_file('data/elon.jpg')
imgElon=resize(imgElon , 0.50)
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)


imgTest = face_recognition.load_image_file('data/ElonMusktest.jpg')
imgTest=resize(imgTest , 0.50)
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)} ',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)


cv2.imshow("eleon mush", imgElon)
cv2.imshow("eleon mush test", imgTest)
cv2.waitKey(0)
