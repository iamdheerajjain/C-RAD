import os
import cv2     # for capturing videos
import math 
import imageio.v3 as iio
import geocoder
import requests
import pandas as pd
from twilio.rest import Client
from geopy.geocoders import Nominatim
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from matplotlib import pyplot as plt 
from skimage.transform import  resize 

count = 0
videoFile = "Accidents.mp4"
cap = cv2.VideoCapture(videoFile)# capturing the video from the given path
if not cap.isOpened():
    raise ValueError(f"Unable to open video file: {videoFile}")
frameRate = cap.get(cv2.CAP_PROP_POS_FPS) #frame rate
x=1
while cap.isOpened():
    frameId = cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame number
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot fetch frame.")
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
        print(f"Saved frame {count} as {filename}")
        count += 1
cap.release()
print ("Done!")

img = plt.imread('0.jpg')   # reading image using its name
plt.imshow(img)

data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()

X = [ ]     # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X) 

y = data.Class
dummy_y = np_utils.to_categorical(y)

image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)

from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X,data_format=None)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

X_train = X_train.reshape(155, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(67, 7*7*512)

train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()

model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

count = 0
videoFile = "Accident-1.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename ="test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

test = pd.read_csv('test.csv')

test_image = []
for img_name in test.Image_ID:
    img = plt.imread('' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

test_image = preprocess_input(test_image, data_format=None)

test_image = base_model.predict(test_image)
test_image.shape

test_image = test_image.reshape(9, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

predictions = model.predict(test_image)

for i in range (0,9):
    if predictions[i][0]<predictions[i][1]:
        print("No Accident")
    else:
        print("Accident")
        
geoLoc = Nominatim(user_agent="GetLoc")
g = geocoder.ip('me')
locname = geoLoc.reverse(g.latlng)
account_sid = 'AC9b22e226e547df02a42fa1f416ef77b7'
auth_token = 'afbdc5b54378547fb092e6a7ced1381c'
client = Client(account_sid, auth_token)

cap = cv2.VideoCapture('Accident-1.mp4')
i=0
flag=0
while(True):
    ret,frame=cap.read()
    if ret==True:
        if predictions[int(i/15)%9][0]<predictions[int(i/15)%9][1]:
            predict="No Accident"
        else:
            predict="Accident"
            flag=1
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                predict,
                (50, 50),
                font, 1,
                (0, 255, 255),
                3,
                cv2.LINE_4)
        cv2.imshow('Frame', frame)
        i=i+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
if flag==1:
    client.messages.create(
                body="Accident detected in "+locname.address,
                from_= '+18777804236',
                to= '+91 8618229536' )
    print("Done")

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
