import cv2
from os import *
import numpy as np

data_path='C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/images/'
onlyfiles=[f for f in listdir(data_path) if path.isfile(path.join(data_path,f))]

Training_Data,Labels=[],[]
for i, files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)
Labels=np.asarray(Labels,dtype=np.int32)
model=cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data),np.asarray(Labels))

face_classifier=cv2.CascadeClassifier('/Users/Dell/Anaconda3/envs/opencv-env/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
def face_detector(img,size=0.5):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
    return img,roi
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    image,face=face_detector(frame) 
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        if result[1]<500:
            conf=(100-int(100*(result[1])/300))
            display_str=str(conf)+'% Confidence it is user'
            cv2.putText(image,display_str,(100,120),cv2.FONT_HERSHEY_COMPLEX,1,(255,250,120),2)

        if(conf>75):
            cv2.putText(image,"UnLocked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Face Cropper',image)
        else:
            cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            cv2.imshow('Face Cropper',image)
    except:
        cv2.putText(image,"Face Not Found",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(image,"Locked",(250,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

        cv2.imshow('Face Cropper',image)
        pass
        
            

    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
                       
print("Complete")
