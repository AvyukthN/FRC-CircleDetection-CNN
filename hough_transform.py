import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os

def hough_transform(img: np.ndarray) -> np.ndarray:
	blurframe = cv2.GaussianBlur(img, (17, 17), 0)
	# ret,thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
	circles = cv2.HoughCircles(blurframe,cv2.HOUGH_GRADIENT,1.2,100,
                            param1=100,param2=30,minRadius=100,maxRadius=1000)

	try:
		circ = circles[0]
	except:
		cv2.imwrite('error_image.png', blurframe)

	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

	return circ 

if __name__ == '__main__':
	# print(os.listdir('./data/ball_dataset'))
	for file in os.listdir('./data/ball_dataset-nopreprocessing'):
		img = cv2.imread(f'./data/ball_dataset-nopreprocessing/{file}', 0)

		start = time.time()
		circ = hough_transform(img)

		x, y, r = circ[0]
		with open('./data/labels/ball_dataset_labels.txt', 'a') as f:
			num = file.split('.')[0]

			x1 = x - r
			y1 = y - r
			w = (x+r) - x1
			h = (y+r) - y1

			f.write(f'{num} $ {x1} {y1} {w} {h}\n')

		print(f'time -> {time.time() - start}')

		# plt.imshow(cimg)
		# plt.show()