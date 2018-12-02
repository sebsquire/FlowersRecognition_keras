'''
Data presented in 5 different folders, one for each class ~800 images per folderA
function creates two .npy files containing image data and labels with roughly even distribution of flower types in each
train = 90%, test = 10%
combines into single list containing 2 arrays. [0] = image data, [1] = one hot encoded class label
'''
import os
import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle


def create_data(folders_dir, img_size):
    data = []      # create full data with (numerical) labels
    for fold in os.listdir(folders_dir):
        # one hot encoding class labels
        if fold == 'daisy': label = [1, 0, 0, 0, 0]
        elif fold == 'dandelion': label = [0, 1, 0, 0, 0]
        elif fold == 'rose': label = [0, 0, 1, 0, 0]
        elif fold == 'sunflower': label = [0, 0, 0, 1, 0]
        elif fold == 'tulip': label = [0, 0, 0, 0, 1]
        # resize and collect data from images
        for img in tqdm(os.listdir(os.path.join(folders_dir, str(fold)))):
            path = os.path.join(folders_dir, fold, img)
            img = cv2.resize(cv2.imread(path, 1), (img_size, img_size))
            data.append([np.array(img), label])
    # data manipulation
    shuffle(data)                               # randomly order data
    training_data = data[:4105]                 # train data
    testing_data = data[4105:]                  # test data
    np.save('train_data.npy', training_data)
    np.save('test_data.npy', testing_data)
    return training_data, testing_data
