import pickle
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import json

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model



def accessCamera() :
  return cv2.VideoCapture(0)

def readImage(path) :
  return cv2.imread(path)

def getSizeOfImg(img) :
  h, w, c = img.shape
  print("Img size (width: {}, height: {}, channels: {})".format(w,h,c))

def getSizeOfVideo(video): 
  width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
  height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
  print("Video size (width: {}, height: {})".format(width, height))

def convertImageToGrayScale(image) :
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # The cv2.cvtColor() function takes two arguments: the first is the color image to be converted, and the second is the color conversion code.

def load_coordinate(label_path) :
  with open(label_path, 'rb') as file :
    label = json.load(file)
    
  return [
    int(math.ceil(label["shapes"][0]['points'][0][0])),
    int(math.ceil(label["shapes"][0]['points'][0][1])),
    int(math.ceil(label["shapes"][0]['points'][1][0])),
    int(math.ceil(label["shapes"][0]['points'][1][1])),
  ]

def preprocess(folder_in, folder_out, coordinates, label) :
  curentImg = cv2.imread(folder_in)
  curentImgRGB = cv2.cvtColor(curentImg, cv2.COLOR_BGR2RGB)    
  
  coords = [0,0,0,0]
  coords[0] = coordinates[0]
  coords[1] = coordinates[1]
  coords[2] = coordinates[2]
  coords[3] = coordinates[3]
  
  sub_faceRGB = curentImgRGB[coordinates[0] : coordinates[0] + coordinates[2], coordinates[1] : coordinates[1] + coordinates[3]]
  
  cv2.imwrite(os.path.join('Cropped_Face_Images', folder_out, 'Long', label.split('.')[0] + '.jpg'), sub_faceRGB)
  
  print("Processing : ", label.split('.')[0] + '.jpg')

def drawBoundingBox(face, image) :
    #  x and y are the coordinates of the top-left corner of the face rectangle
    #  w is the width of the rectangle, and h is the height of the rectangle.
    for (x, y, w, h) in face:
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # The cv2.rectangle() is called to draw a rectangle on the original image (img) using the coordinates (x,y) and dimensions(w,h) of the detected face.
    # (0, 255, 0) means the rectangle is drawn with a green color (r,g,b)
    # the number 4 is the thickness of rectangle

def collectFace(vid) :
  count = 0
  while True:
    result, video_frame = vid.read() 
    if result is False:
      break
    
    # cv2.imwrite("Data_Raw\\Long\\z51985795_roi{}.jpg".format(count), video_frame)
    # cv2.imwrite("Data_Raw\\Phuc\\z51985795_roi{}.jpg".format(count), video_frame)
    # cv2.imwrite("Data_Raw\\Quoc\\z51985795_roi{}.jpg".format(count), video_frame)
    count += 1
      
    cv2.imshow("Collecting......", video_frame) 
    time.sleep(0.2)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  vid.release()
  cv2.destroyAllWindows()
  
def Detect_And_Identify(vid) :
  quoc = load_model('Quoc.h5')
  face_recognition = load_model('Face_Recognition.keras')
  
  with open("ResultsMap.pkl", 'rb') as file:
   result_map = pickle.load(file)

  
  while True :
    result, frame = vid.read()
    if result is False:
      break
    frame = frame[50 : 500, 50 : 500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    # long_yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    yhat = quoc.predict(np.expand_dims(resized/255, 0))
    # print(yhat[1][0])
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5 :
      # Control the main rectangle
      cv2.rectangle(
        frame,
        tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
        tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
        (255,0,0),
        2 
      )  
      
      resize = cv2.resize(
        frame[
          np.multiply(sample_coords[1], 450).astype(int) : np.multiply(sample_coords[3], 450).astype(int),
          np.multiply(sample_coords[0], 450).astype(int) : np.multiply(sample_coords[2], 450).astype(int)
        ],  
        (120,120)
      )
      
      name = face_recognition.predict(np.expand_dims(resize/255, 0), verbose=0)
      
      # cv2.imwrite(
      #   os.path.join('Data_Raw', 'Long', f'Long.{count}.jpg'), 
      #   frame[
      #     np.multiply(sample_coords[1], 450).astype(int) : np.multiply(sample_coords[3], 450).astype(int),
      #     np.multiply(sample_coords[0], 450).astype(int) : np.multiply(sample_coords[2], 450).astype(int)
      #   ]          
      # )
      
      # Control the text rendered
      cv2.putText(
        frame, 
        result_map[np.argmax(name)],
        tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[0, -5])),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2,
        cv2.LINE_AA
      )
      
    cv2.imshow("My Face Detection Project", frame) 
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  vid.release()
  cv2.destroyAllWindows()



video_stream = accessCamera()
# collectFace(video_stream)
Detect_And_Identify(video_stream)


