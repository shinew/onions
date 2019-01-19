import io
import os
import sys

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Edge detection
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = sys.argv[1] if len(sys.argv)>1 else 'onion.jpg'

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)

#print('Labels:')
#for label in labels:
#    print(label.description)

def is_onion(response):
    labels = response.label_annotations
    print(labels)
    return True
    for label in labels:
        if label.description == 'onion' and \
           label.score > 0.7:
            return True
    return False

import circle_detector

img = cv2.imread(file_name)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

if is_onion(response):
    print('onion')
    cv2.putText(img,"Onion", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    detector = circle_detector.CircleDetector()
    circle = detector.detect(file_name)
    cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(circle[0],circle[1]),2,(0,0,255),3)
else:
    print('not onion')
    cv2.putText(img,"NOT Onion", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

plt.imshow(img)
plt.show()
