import cv2
import matplotlib.pyplot as plt
import os

img_hash = {}

with open('./data/labels/ball_dataset_labels.txt', 'r') as f:
    arr = f.read().split('\n')

    for i in range(len(arr)):
        temp = arr[i]
        temp = temp.split('$')

        temp0 = temp[0].strip()
        temp1 = temp[1].strip()

        try:
            img_hash.update({temp0: [int(float(num)) for num in temp1.split(' ')]})
        except:
            print(i)

print(img_hash)

count = 0
biggest_rad = float('-inf')

for file in os.listdir('./data/ball_dataset-nopreprocessing'):
    img = cv2.imread(f'./data/ball_dataset-nopreprocessing/{file}')

    file_num = file.split('.')[0]

    epsilon = 15

    # SWITCH TO OVERLAYING WHATS FOUND IN THE BOUNDING BOX ONTO A BLACK IMAGE

    x, y, w, h = img_hash[file_num]
    # img = cv2.rectangle(img, (x-epsilon, y-epsilon), (x+w+epsilon, y+h+epsilon), (0, 0, 0), 2)
    
    if w > biggest_rad:
        biggest_rad = w

    # to_save = img[y-epsilon:y+h+epsilon, x-epsilon:x+w+epsilon]
    # cv2.imwrite(f'./data/ball_dataset-preprocessed/{count}.png', to_save)

    count += 1

    print(f'preprocessed => image file {count}')
    # plt.imshow(img)
    # plt.show()

print(biggest_rad)