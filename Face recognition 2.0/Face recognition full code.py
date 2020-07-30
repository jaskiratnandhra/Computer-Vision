import cv2
import numpy as np
from keras.models import load_model
from PIL import Image


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

train_dir='C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/train_images'
train_generator=train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=10,class_mode='binary')
test_dir='C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/test_images'
validation_generator=train_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=10,class_mode='binary')

history=model.fit_generator(train_generator,steps_per_epoch=10,epochs=15,validation_data=validation_generator,validation_steps=5,verbose=2)
model.save('C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/model.h5')


#Prediction

model=load_model("model.h5")
path="C:/Users/Dell/Desktop/study/Face reco/face recon 3.0/train_images/mom/mom233.png"
img=Image.open(path)
img=img.resize((150,150))
x=np.array(img)
print(x.shape)
x=np.expand_dims(x, axis=0)
images=np.vstack([x])

classes=model.predict(x)
print(classes[0])
if(classes[0]>0):
    print("mom")
else:
    print("jaskirat")
