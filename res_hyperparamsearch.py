'''
Multi class classification of flower types using CNN - including a residual layer
Model attains 79% accuracy in 30 epochs using:
 - Adadelta with lr=0.25, weight_decay=1e-6, keep_rate=0.8, batch_size=32
Training is also noticeably quicker with residual layers although overfit was prevalent.

Info on residual layers:
    ResNet-50:
        - Deep Residual Learning for Image Recognition - https://arxiv.org/pdf/1512.03385.pdf
    ResNeXt-50 32x4d
        - Aggregated Residual Transformations for Deep Neural Networks - https://arxiv.org/pdf/1611.05431.pdf

To run this you will need to change:
 - directory of image_folders_dir - point it to the folder containing all flower type folders

Python 3.6.7, Tensorflow 1.12.0, Keras 2.2.4
'''
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input, BatchNormalization, add, Activation
from keras.optimizers import adadelta
from keras.models import Model

# image_folders_dir is location of folders containing images of various flower types
image_folders_dir = 'C:\\Users\squir\Dropbox\ML Projects\Kaggle\Flowers Recognition\\flowers'

IMG_SIZE = 128                          # resize image to this height and width
num_classes = 5                         # different flower types
epochs = 30                             # number of times model sees full data

lr = [0.25, 0.5]                  # learning rate options for hyperparam tuning
dropout_keep_rate = [0.8, 0.9]     # dropout options for hyperparam tuning
batch_size = [32, 64, 96]              # batch size options for hyperparam tuning

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
4321 images are now:
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
with open('res_layer_results_file2.csv', 'w') as f:
    f.write('lr,keep rate,batch size,val_accuracy,epoch\n')

# ---------- MODELLING AND TESTING ----------
# for loops for grid search of hyperparameter options
for i in lr:
    for j in dropout_keep_rate:
        for k in batch_size:
            # to save well performing models
            MODEL_NAME = 'flowers-{}-{}-{}-{}-{}.model'\
                .format('lr'+str(i), 'dr'+str(j), 'bs'+str(k), '5-layer', 'resnet-basic')

            # set input
            x = Input(shape=(128, 128, 3))

            # ----- shortcut connection path -----
            shortcut_y = x
            shortcut_y = Conv2D(512, (32, 32), strides=(32, 32), padding='valid')(shortcut_y)
            shortcut_y = Dropout(0.2)(shortcut_y)
            shortcut_y = BatchNormalization()(shortcut_y)

            # ----- convolutional path ------
            conv_y = x
            # layer 1
            conv_y = (Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))(conv_y)
            conv_y = (MaxPool2D(pool_size=(2, 2)))(conv_y)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            # layer 2
            conv_y = (Conv2D(64, (3, 3), input_shape=(64, 64, 3), activation='relu'))(conv_y)
            conv_y = (MaxPool2D(pool_size=(2, 2)))(conv_y)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            # layer 3
            conv_y = (Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))(conv_y)
            conv_y = (MaxPool2D(pool_size=(2, 2)))(conv_y)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            # layer 4
            conv_y = (Conv2D(256, (3, 3), input_shape=(16, 16, 3), activation='relu'))(conv_y)
            conv_y = (MaxPool2D(pool_size=(2, 2)))(conv_y)
            conv_y = BatchNormalization()(conv_y)
            conv_y = Activation('relu')(conv_y)
            # layer 5
            conv_y = (Conv2D(512, (3, 3), input_shape=(8, 8, 3), activation='relu'))(conv_y)
            conv_y = BatchNormalization()(conv_y)

            # combine convolutional and shortcut connection
            y = add([shortcut_y, conv_y])
            y = Activation('relu')(y)

            # fully connected
            y = Flatten()(y)
            y = Dense(128, activation='relu')(y)
            y = Dropout(j)(y)
            predictions = Dense(5, activation='softmax')(y)
            model = Model(inputs=x, outputs=predictions)

            adadelt = adadelta(lr=i, decay=1e-6)
            model.compile(loss='categorical_crossentropy', optimizer=adadelt, metrics=['accuracy'])
            out = model.fit(X_train, Y_train,
                      batch_size=k,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_valid, y_valid))

            # show model testing parameters
            print(out.params)

            # recording testing accuracy results
            max_val_acc = max(out.history['val_acc'])
            max_ep = [a+1 for a, b in enumerate(out.history['val_acc']) if b == max_val_acc]
            with open('res_layer_results_file2.csv', 'a') as f:
                f.write('{},{},{},{},{}\n'.format(i,
                                                  j,
                                                  k,
                                                  max_val_acc,
                                                  max_ep[0]))

            # un-comment below to save model
            # model.save(MODEL_NAME)
