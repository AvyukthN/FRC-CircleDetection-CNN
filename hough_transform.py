import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random
from networktables import NetworkTablesInstance
from networktables import NetworkTables 

def hough_transform(img: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	image_copy = gray
	mask = np.zeros(image_copy.shape, dtype="uint8")
	# test_mask = image_copy.copy()
	for c in contours:
		# Defining a condition to draw the contours, a number larger than/smaller than or between ranges
		thresh = 500
		if cv2.contourArea(c) > thresh:
			x, y, w, h = cv2.boundingRect(c)
			# cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 2)
			if ((max(w,h) / min(w, h)) - 1) < 0.2:
				# cv2.drawContours(image_copy, [c], 0, (255,255,255), 3)
				# cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), 2)

				mask[y:y+h, x:x+w] = image_copy[y:y+h, x:x+w]
			# cv2.rectangle(test_mask, (x, y), (x+w, y+h), (255, 255, 255), 2)
	
	# cnts_img = cv2.drawContours(image=img.copy(), contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
	
	# cv2.imwrite('./contour_mask.png', cnts_img)
	# cv2.imwrite('./contour_rects.png', test_mask)

	blurframe = cv2.GaussianBlur(mask, (17, 17), 0)
	circles = cv2.HoughCircles(blurframe,cv2.HOUGH_GRADIENT,1.2,100, param1=100,param2=30,minRadius=0,maxRadius=200)

	return circles, mask

if __name__ == '__main__':

	# try different backedns with cv2.CAP_DSHOW	because 2 webcams might be too much or smthing
	# remove if not working to test
	vid = cv2.VideoCapture(1, cv2.CAP_ANY)

	luffy_face = cv2.imread('./luffy_face.jpg')
	luffy_face = cv2.resize(luffy_face, (100, 100))

	# # start NetworkTables
	# ntinst = NetworkTablesInstance.getDefault()
	NetworkTables.initialize(server='roborio-*INSERT NUMBER*-frc.local')
    # Name of network table - this is how it communicates with robot. IMPORTANT
	dash_table = NetworkTables.getTable('SmartDashboard')

	# start = time.time()	
	while True:
		ret, frame = vid.read()

		if ret:
			# img = cv2.resize(frame, (1500, 1000))
			img = frame
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

			'''
			FOR BLUE HSV DETECTION ONLY
			# BLUE HSV
			# lower_blue = np.array([100, 50, 120])
			# upper_blue = np.array([140, 255, 255])
			# mask = cv2.inRange(hsv, lower_blue, upper_blue)

			# where there is something in the frame and the mask is true
			# bitmask = cv2.bitwise_and(img, img, mask=mask)
			'''

			# RED HSV
			lower_red1 = np.array([0, 130, 100])
			upper_red1 = np.array([10, 255, 255])
			lower_red2 = np.array([160, 100, 100])
			upper_red2 = np.array([179, 255, 255])
			
			lower_mask = cv2.inRange(hsv, lower_red1, upper_red1)
			upper_mask = cv2.inRange(hsv, lower_red2, upper_red2)

			full_mask = lower_mask + upper_mask

			bitmask = cv2.bitwise_and(img, img, mask=full_mask)

			circles, rect_mask = hough_transform(bitmask)

			try:
				circ = circles[0]
				descriptors = []
				
				circles = np.uint16(np.around(circles))
				for i in circles[0,:]:
					# draw the outer circle
					cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
					# draw the center of the circle
					cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)

					# centerx = i[0]
					# centery = i[1]
					# ov_w, ov_h, _ = luffy_face.shape

					# start_x = centerx - (ov_w//2)
					# start_y = abs(centery - (ov_h//2))

					# end_x = start_x + ov_w
					# end_y = start_y + ov_h

					# frame[start_y:end_y, start_x:end_x] = luffy_face

					descriptors.append({"center": (i[0], i[1]), "diameter": 2*i[2]})

				biggest_d = float('-inf')
				for hash in descriptors:
					diameter = hash["diameter"]

					if diameter > biggest_d:
						biggest_d = diameter 

				distance = (9.5 * 8) / biggest_d

				dash_table.putNumber('closest_ball_dist', distance)
				# print(f'DISTANCE OF CLOSEST BALL -> {distance}')
			except:
				pass
			
			# if time.time() - start > 10:
			# 	break

			cv2.imshow('circ_detection', frame)
			cv2.imshow('bitmask', bitmask)
			cv2.imshow('contour_mask', rect_mask)
			cv2.waitKey(1)