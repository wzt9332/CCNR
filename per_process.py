import cv2
import numpy as np

class_n = 9
crop_size = [16, 16]  # patch size
crop_num = [16, 16]  # step num
data_root = './data/hongkong'
file_name = data_root + '/train.txt'
with open(file_name) as f:
    idx_list = f.read().splitlines()

class_num = np.zeros(class_n).tolist()
for name in idx_list:
    img = cv2.imread(data_root + '/SegmentationClassAug/{}.png'.format(name), cv2.IMREAD_UNCHANGED)
    for i in range(crop_num[0]):
        for j in range(crop_num[1]):
            img_crop = img[i * crop_size[0]: i * crop_size[0]+crop_size[0], j * crop_size[1]: j * crop_size[1]+crop_size[1]]
            sort = sorted([(np.sum(img_crop == w), w) for w in set(img_crop.flat)])
            class_num[sort[0][1]] = int(class_num[sort[0][1]] + 1)

print('regional class statistics: ', class_num)




