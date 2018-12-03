'''
Multi class classification of flower types using CNN
Base model:
  - 5 convolutional layers w/ max pooling proceeding each of the first 4.
  - 1 fully connected layer
  - 1 output layer (softmax)
Model attains 81% accuracy in 30 epochs using:
 - Adadelta with lr=0.5, weight_decay=1e-6, keep_rate=0.9, batch_size=16

To run this you will need to change:
 - directory of image_folders_dir - point it to the folder containing all flower type folders

Python 3.6.7, Tensorflow 1.12.0, Keras 2.2.4
'''
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import adadelta
from keras.layers.core import Dense, Dropout

# image_folders_dir is location of folders containing images of various flower types
image_folders_dir = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Flowers Recognition\\flowers'

IMG_SIZE = 128                          # resize image to this height and width
num_classes = 5                         # different flower types
epochs = 30                             # number of times model sees full data

lr = [0.5, 1, 2]                        # learning rate options for hyperparam tuning
dropout_keep_rate = [0.9, 0.95, 1]      # dropout options for hyperparam tuning
batch_size = [16, 32, 64]               # batch size options for hyperparam tuning

# Ask user to load or process - for first time need to process but subsequently can load data
# UNLESS IMG_SIZE is changed
print('Load pre-existing preprocessed data for training (L) or preprocess data (P)?')
decision1 = input()
if decision1 == 'P' or decision1 == 'p':
    from preprocessing import create_data
    train_data, test_data = create_data(image_folders_dir, IMG_SIZE)
elif decision1 == 'L' or decision1 == 'l':
    if os.path.exists('train_data.npy'):
        train_data = np.load('train_data.npy')
        test_data = np.load('test_data.npy')
    else:
        raise Exception('No preprocessed data exists in path, please preprocess some.')
else:
    raise Exception('Please retry and type L or P')

'''
Images are now:
IMG_SIZE*IMG_SIZE*RGB attached to one hot class label flower type and ordered randomly
Data is comprised of a list containing: [0]: image data, [1]: class label
'''
# create image (arrays) and label (lists) for use in models
X_train = np.array([item[0] for item in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_train = np.array([item[1] for item in train_data])
x_valid = np.array([item[0] for item in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y_valid = np.array([item[1] for item in test_data])

X_train = X_train / 255                 # normalising
x_valid = x_valid / 255                 # normalising

# create results file
with open('results_file.csv', 'w') as f:
    f.write('lr,keep rate,batch size,val_accuracy,epoch\n')

# ---------- MODELLING AND TESTING ----------
# for loops for grid search of hyperparameter options
for i in lr:
    for j in dropout_keep_rate:
        for k in batch_size:
            # to save well performing models
            MODEL_NAME = 'flowers-{}-{}-{}-{}-{}.model'\
                .format('lr'+str(i), 'dr'+str(j), 'bs'+str(k), '5-layer', 'resnet-basic')

            model = Sequential()
            # layer 1
            model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            # layer 2
            model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            # layer 3
            model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            # layer 4
            model.add(Conv2D(256, (3, 3), input_shape=(16, 16, 3), activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            # layer 5
            model.add(Conv2D(512, (3, 3), input_shape=(8, 8, 3), activation='relu'))
            # fully connected
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(j))
            model.add(Dense(5, activation='softmax'))

            adadelt = adadelta(lr=i, decay=1e-6)
            model.compile(loss='categorical_crossentropy', optimizer=adadelt, metrics=['accuracy'])

            out = model.fit(X_train,
                            Y_train,
                            batch_size=k,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_valid, y_valid))

            # show model testing parameters
            print(out.params)

            max_val_acc = max(out.history['val_acc'])
            max_ep = [a+1 for a, b in enumerate(out.history['val_acc']) if b == max_val_acc]
            with open('results_file.csv', 'a') as f:
                f.write('{},{},{},{},{}\n'.format(i,
                                                  j,
                                                  k,
                                                  max_val_acc,
                                                  max_ep[0]))

            # un-comment below to save model
            # model.save(MODEL_NAME)
