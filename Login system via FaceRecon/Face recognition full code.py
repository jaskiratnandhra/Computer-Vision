import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
'''
#Formatiom of Image data set
 
camera = cv2.VideoCapture(0)
for i in range(500):
    return_value, image = camera.read()
    #image=image.resize(150,150)
    cv2.imwrite('C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/train_images/mom/mom'+str(i)+'.png', image)
    print('click')
del(camera)



    
#Training of model 

import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras.optimizers import RMSprop

model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(100,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(200,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(250,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  
  tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1/255)
test_datagen=ImageDataGenerator(rescale=1/255)

train_dir='C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/train_images'
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=10,class_mode='binary')
test_dir='C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/test_images'
validation_generator=train_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=10,class_mode='binary')

history=model.fit_generator(train_generator,steps_per_epoch=10,epochs=15,validation_data=validation_generator,validation_steps=5,verbose=2)
model.save('C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/model.h5')

'''
#Prediction


model=load_model("model.h5")
face_=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    faces=face_.detectMultiScale(gray,1.3,5)
    #if faces is():
     #  return None
    for(x,y,w,h) in faces:
        cropped_face=img[y:y+h,x:x+w]
        return cropped_face


cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(196,196))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2HSV)
        
        file_name_path='C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/image/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(196,196),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Croper',face)
    else:
        print('face not found kindly allign your face')
        pass
    
       
    if cv2.waitKey(1)==27 or count==10:
        
        break
cap.release()
cv2.destroyAllWindows()
l=[]

for i in range(1,11):
    path='C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/image/'+str(i)+'.jpg'
#path='C:/Users/Jaskirat/Desktop/study/ML codes/Face reco/face recon 3.0/train_images/jaskirat/j.jpg'
img=Image.open(path)
img=img.resize((150,150))
x=np.array(img)
print(x.shape)
x=np.expand_dims(x, axis=0)
images=np.vstack([x])

classes=model.predict(x)
print(classes[0])
if(classes[0]>0):
    l+=["shivam"]
    #print("shivam")
else:
    l+=["jaskirat"]
    #print("jaskirat")


s=0
j=0
for i in l:
    if i=="shivam":
        s+=1
    else:
        j+=1
if(s>j):
    #print("Welcome Shivam")

    n="Shivam"
else:
    #print("WEcome Jaskirat")
    n="Jaskirat"
from tkinter import *


def click():
    a.destroy() 
    g=Tk()
    g.title('Success')
    g.geometry('300x100')
    r=Label (g, text="Login Success.", font="none 24 bold")
    r.config(anchor=CENTER)
    r.pack()
    g.mainloop()
from tkinter import *

a=Tk()
a.title('tk')
a.geometry('300x200')
te="Welcome "+n
z=Label (a, text=te, font="none 24 bold")
e=Label (a, text="Enter Your Pin here", font="none 18 bold")
z.config(anchor=CENTER)
e.config(anchor=CENTER)
z.pack()
e.pack()
reg=Label(a,text="PIN:")
regtxt=Entry(a,width=10)
#reg.config(anchor=CENTER)
#regtxt.config(anchor=CENTER)
reg.pack()
regtxt.pack()
b=Button(a,text='Submit',command=click)
b.pack()
a.mainloop()


