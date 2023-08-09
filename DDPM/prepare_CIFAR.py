import os

import cv2
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


loc_1 = './datasets/train_cifar10/'

if os.path.exists(loc_1) == False:
    os.mkdir(loc_1)


# each file contains 10,000 images
def cifar10_img(file_dir):
    for i in range(1, 6):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')

        for j in range(10000):
            img = np.reshape(data_dict[b'data'][j], (3, 32, 32))
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_name = loc_1 + str(data_dict[b'labels'][j]) + str((i) * 10000 + j) + '.jpg'
            cv2.imwrite(img_name, img)

        print(data_name + ' is done')

    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10000):
        img = np.reshape(test_dict[b'data'][m], (3, 32, 32))
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_name = loc_1 + str(test_dict[b'labels'][m]) + str(m) + '.jpg'
        cv2.imwrite(img_name, img)
    print(test_data_name + ' is done')
    print('Finish transforming to image')


if __name__ == '__main__':
    file_dir = './datasets/cifar-10-batches-py'
    cifar10_img(file_dir)
