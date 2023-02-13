import cv2
import numpy as np
import glob
import os

frameSize = (640, 480)

out = cv2.VideoWriter('/home/ubuntu/Desktop/iGibson_study/igibson_dataset/01/video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, frameSize)

directory = "/home/ubuntu/Desktop/iGibson_study/igibson_dataset/01/rgb/"
for i in range(1, 201):
    f = directory + str(i) + ".png"
    img = cv2.imread(f)
    out.write(img)

out.release()
