import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random

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

# print(img_hash)

count = 0

# biggest radius found by iterating through data previously
biggest_rad = 1420 # float('-inf')

for file in os.listdir('./data/ball_dataset-nopreprocessing'):
    color_img = cv2.imread(f'./data/ball_dataset-nopreprocessing/{file}')
    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    file_num = file.split('.')[0]

    epsilon = 15

    # SWITCH TO OVERLAYING WHATS FOUND IN THE BOUNDING BOX ONTO A BLACK IMAGE

    x, y, w, h = img_hash[file_num]

    if (x - epsilon < 0) or (y - epsilon < 0):
        epsilon = 0
    
    if (x < 0):
        x = 0
    if (y < 0):
        y = 0

    mult = random.uniform(0.5, 2)

    underlay_size_x = int((w * mult) + w)
    underlay_size_y = int((h * mult) + h)
    black_img = np.zeros((underlay_size_y, underlay_size_x))
    overlay = img[y-epsilon:y+h+epsilon, x-epsilon:x+w+epsilon]

    # np.shape -> (rows, columns)
    x_range = (underlay_size_x // 2) - (overlay.shape[1] // 2)
    y_range = (underlay_size_y // 2) - (overlay.shape[0] // 2)

    shifter_x = random.randint(-1*x_range, x_range)
    shifter_y = random.randint(-1*y_range, y_range)

    x_offset = x_range + shifter_x
    y_offset = y_range + shifter_y

    # too lazy to deal with shifter values that put the overlay image out of range so i just while looped till they didnt -> ingenious problem solving skill
    # it doesnt matter its a preprocesssing script that runs like once before we start training the CNN
    # technically im being more efficient by choosing the option that takes less time -> problem solving 1000
    while (x_offset + overlay.shape[0] > underlay_size_x) or (y_offset + overlay.shape[1] > underlay_size_y):
        shifter_x = random.randint(-1*x_range, x_range)
        shifter_y = random.randint(-1*y_range, y_range)

        x_offset = x_range + shifter_x
        y_offset = y_range + shifter_y

    try:
        black_img[x_offset:x_offset+overlay.shape[0], y_offset:y_offset+overlay.shape[1]] = overlay
    except Exception as e:
        print('')

        print('offsets')
        print(x_offset, y_offset)

        print('overlay and underlay dims')
        print(black_img.shape)
        print(overlay.shape)

        print('file-number')
        print(file_num)

        print(e)

        break
    
    # randomly blur
    if random.randint(1, 5) == 5:
        dims = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        kernel_dim = dims[random.randint(0, len(dims)-1)]

        black_img = cv2.GaussianBlur(black_img, (kernel_dim, kernel_dim), 10)

    # img = cv2.rectangle(img, (x-epsilon, y-epsilon), (x+w+epsilon, y+h+epsilon), (0, 0, 0), 2)
    
    # if w > biggest_rad:
    #     biggest_rad = w

    cv2.imwrite(f'./data/ball_dataset-preprocessed/{count}.png', black_img)

    count += 1

    print(f'preprocessed => image file {count}')
    # plt.imshow(img)
    # plt.show()