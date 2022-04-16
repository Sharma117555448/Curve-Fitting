# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:33:05 2022

@author: sharm
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
 
# Captures the points from the input video    
Parser = argparse.ArgumentParser()
Parser.add_argument('--pathToVideo', default='ball_video2.mp4', help=', Default: ./video/ball_video1.mp4.mp4')
Args = Parser.parse_args()
video = Args.pathToVideo
print("\n")


cap = cv2.VideoCapture(video)

# Function for matrix transform
def transformer(cord):
    shape = len(cord)

    x_tilde = np.zeros(shape=(shape, 3))
    y_tilde = np.zeros(shape=(shape, 1))

    for i, point in enumerate(cord):
        x_tilde[i] = [np.power(point[0], 2), point[0], 1]
        y_tilde[i] = [point[1]]
    return x_tilde, y_tilde

# Function to determine the mean error
def meanError(points,matrix):
    error = 0
    for point in points:
        y = matrix[0]*(np.power(point[0],2)) + matrix[1]*point[0] + matrix[2]
        error = error + abs(y-point[1])

    e = error/len(points) 
    return e

# Function to converts given points to fit the curve using LS algorithm
def leastSquare(c):
    ax, ay = transformer(c)
    axT = np.transpose(ax)


    first = np.matmul(axT, ax)
    first_inv = np.linalg.pinv(first)

    second = np.matmul(axT, ay)
    before_transpose = np.matmul(first_inv, second)

    coefficient_matrix = before_transpose.transpose()[0]
    error = meanError(c, coefficient_matrix)
    print("Least Square Coefficients: \n", coefficient_matrix, "\n with error =", error,"\n")
    transformedY = np.matmul(ax, coefficient_matrix)
    return transformedY

# Function for plotting the graph
def plotPlots(x_coord, y_coord, yfinal, label, title):
    plt.scatter(x_coord, y_coord, label='Data points', color='red')
    plt.plot(x_coord, yfinal, label=label, color='black', linewidth=2)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


coordinates = []
try:
    while(cap.isOpened()):
        ret, frame = cap.read()
 
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # dimensions =gray.shape
        # print(dimensions) (1676, 2400)

        #cv2.imshow('Video in Grayscale', gray)
        
        # Detect countours
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for countour in contours:
            # Compute the center of the contour
            M = cv2.moments(countour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        coordinates.append((cx, thresh.shape[1]-cy))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except:
    x, y = zip(*coordinates)
    print("Output for the sensor with noise \n")
    ls = leastSquare(coordinates)
    plotPlots(x, y, ls, "LS Fitting", "LS fitting for Sensor with noise ")
    plt.savefig("sensor_with_noise.png", bbox_inches='tight')    
    
cap.release()
cv2.destroyAllWindows()
    