import cv2
import numpy as np

vid = cv2.VideoCapture('./data/red_ball_long.MOV')

count = 0

while True:
    ret, img = vid.read()
    if ret:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv = cv2.GaussianBlur(hsv, (5, 5), 1)

        lower= np.array([0, 150, 0])
        upper = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        # where there is something in the frame and the mask is true
        bitmask = cv2.bitwise_and(img, img, mask=mask)

        # mask = cv2.GaussianBlur(mask, (5, 5), 1)

        cv2.imwrite(f'./data/ball_dataset/{count}.png', mask)
        
        count += 1

        cv2.imshow('mask', mask)
        cv2.imshow('bitmask', bitmask)
        cv2.waitKey(5)
    else:
        break