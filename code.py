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
file_name = '/Users/shine/Downloads/onion3.jpg'

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
    for label in labels:
        if label.description == 'onion' and \
           label.score > 0.7:
            return True
    return False

if is_onion(response):
    sys.exit(0)
