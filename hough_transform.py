import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

def hough_transform(og_img: np.ndarray, img: np.ndarray) -> np.ndarray:
	# h, w = img.shape

	# maxh = 0.25 * h
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	image_copy = gray
	mask = np.zeros(image_copy.shape, dtype="uint8")
	for c in contours:
		# Defining a condition to draw the contours, a number larger than/smaller than or between ranges
		thresh = 500
		if cv2.contourArea(c) > thresh:
			x, y, w, h = cv2.boundingRect(c)
			# cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 2)
			if ((max(w,h) / min(w, h)) - 1) < 0.2:
				# cv2.drawContours(image_copy, [c], 0, (255,255,255), 3)
				# cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 2)

				# plt.imshow(image_copy[y:y+h, x:x+w])
				# plt.show()

				mask[y:y+h, x:x+w] = image_copy[y:y+h, x:x+w]

	# cv2.imwrite('./contour_rects_mask.png', mask)
	# cv2.imwrite('./contour_rects.png', image_copy)

	# gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# edges = cv2.Canny(gray, 100, 5)
	blurframe = cv2.GaussianBlur(mask, (17, 17), 0)
	# ret,thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
	circles = cv2.HoughCircles(blurframe,cv2.HOUGH_GRADIENT,1.2,100, param1=100,param2=30,minRadius=0,maxRadius=200)
	print(circles)
	# circles = cv2.HoughCircles(blurframe,cv2.HOUGH_GRADIENT,1.2,100,
    #                         param1=100,param2=30,minRadius=100,maxRadius=1000)

	return circles


if __name__ == '__main__':
	# print(os.listdir('./data/ball_dataset'))
	# filepath = './data/balls.jpg'
	# start = time.time()
	# circ_img = hough_transform(cv2.imread('./data/balls.jpg'))
	# print(f'time -> {time.time() - start}')
	# cv2.imwrite('out.png', circ_img)
	# filepath = './field.JPG'
	# for file in os.listdir('./data/ball_dataset-nopreprocessing'):

	# inp_img = cv2.imread(filepath)

	vid = cv2.VideoCapture(0)

	while True:
		ret, frame = vid.read()
		if ret:
			# img = cv2.resize(frame, (1500, 1000))
			img = frame
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			# hsv = cv2.GaussianBlur(hsv, (5, 5), 1)

			# lower= np.array([0, 130, 150])
			# upper = np.array([255, 255, 255])

			# lower = np.array([20, 100, 100])
			# upper = np.array([180, 255, 255])

			# lower_blue = np.array([60, 35, 140])
			# upper_blue = np.array([180, 255, 255])

			lower_blue = np.array([100, 50, 0])
			upper_blue = np.array([140, 255, 255])

			mask = cv2.inRange(hsv, lower_blue, upper_blue)
			# where there is something in the frame and the mask is true
			bitmask = cv2.bitwise_and(img, img, mask=mask)

			# if random.randint(0, 100) == 10:
			
			circles = hough_transform(frame, bitmask)

			try:
				circ = circles[0]
				# cv2.imwrite('./good.png', bitmask)
				descriptors = []

				circles = np.uint16(np.around(circles))
				for i in circles[0,:]:
					# draw the outer circle
					cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
					# draw the center of the circle
					cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

					descriptors.append({"center": (i[0], i[1]), "diameter": 2*i[2]})

					# cv2.imwrite('./out-detected.png', circ)

					# print(descriptors)

				biggest_d = float('-inf')
				for hash in descriptors:
					diameter = hash["diameter"]

					if diameter > biggest_d:
						biggest_d = diameter 

				distance = (9.5 * 8) / biggest_d
				# print(f'DISTANCE OF CLOSEST BALL -> {distance}')
				# cv2.imwrite('good.png', bitmask)
			except:
				# cv2.imwrite('bitmask.png', bitmask)
				pass

			cv2.imshow('vision', img)
			cv2.waitKey(1)
	
	# plt.imshow(img)
	# plt.show()

	# mask = cv2.GaussianBlur(mask, (5, 5), 1)

	# start = time.time()
	# circ = hough_transform(img)

	# x, y, r = circ[0]
	# with open('./data/labels/ball_dataset_labels.txt', 'a') as f:
	# 	num = file.split('.')[0]

	# 	x1 = x - r
	# 	y1 = y - r
	# 	w = (x+r) - x1
	# 	h = (y+r) - y1

	# 	f.write(f'{num} $ {x1} {y1} {w} {h}\n')

	# print(f'time -> {time.time() - start}')