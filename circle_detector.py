import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class CircleDetector:
    def __init__(self):
        self.resize_x = 300

    def detect(self, path):
        img = cv2.imread(path,0)
        img = cv2.medianBlur(img,5)
        height, width = img.shape[:2]

        ratio = max(height//self.resize_x, 1)
        small = cv2.resize(img, (int(width / ratio), int(height / ratio)), interpolation = cv2.INTER_CUBIC)
        circles = cv2.HoughCircles(small,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=30,minRadius=0,maxRadius=0)
        circles = np.uint16(np.around(circles))
        return (circles[0,0] * ratio).astype(int)

if __name__ == '__main__':
    img = cv2.imread('onion2.jpg',0)
    img = cv2.medianBlur(img,5)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    plt.imshow(cimg)
    plt.show()

