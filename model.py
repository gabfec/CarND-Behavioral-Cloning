import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers import *

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


DATA_SET = 'data'
steering_corrections = {
    'center': 0,
    'left': 0.35,
    'right': -0.35
}
BATCH_SIZE = 32


def get_samples(data_dir):
    csv_file = data_dir + '/driving_log.csv'
    lines = []
    with open(csv_file) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines


all_samples = get_samples(DATA_SET)
train_samples, validation_samples = train_test_split(all_samples, test_size=0.2)


def augment(sample):
    augmented_samples = []
    for idx, camera in enumerate(steering_corrections):
        image = cv2.imread(DATA_SET + '/IMG/' + sample[idx].split('/')[-1])

        center_angle = float(sample[3])
        angle = center_angle + steering_corrections[camera]

        # Random change the brighness
        #yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        #yuv[:, :, 0] = yuv[:, :, 0] + random.uniform(-20, 20)
        #image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # Random shift the image
        #tx = random.uniform(-20, 20)
        #ty = random.uniform(-10, 10)
        #M = np.float32([[1, 0, tx], [0, 1, ty]])
        #image = cv2.warpAffine(image, M, (320, 160))
        #angle += tx * 0.01

        augmented_samples.append((image, angle))

        # Mirror the image (flip around Y axis)
        flipped_image = cv2.flip(image, 1)
        augmented_samples.append((flipped_image, -angle))


    return augmented_samples


def generator(samples, batch_size):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                augmented_samples = augment(sample)

                images.extend([image for image, angle in augmented_samples])
                angles.extend([angle for image, angle in augmented_samples])

            yield shuffle(np.array(images), np.array(angles))


def show_stats(hist):
    '''
    Plot the training and validation loss for each epoch
    '''
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def create_model():
    '''
    This model is based on the Nvidia paper:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    '''
    model = Sequential()

    # Normalise
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

    # Crop 70 pixels from top (sky & trees) and 25 from bottom (car hood)
    model.add(Cropping2D(cropping=((70, 24), (0, 0))))

    # Convolutional layers with kernel 5*5
    model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))

    # Convolutional layers with kernel 3*3
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    model.add(Flatten())

    # Add a dropout layer to overcome overfitting
    model.add(Dropout(0.5))

    # Fully connected layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    # The output of the final layer is the steering prediction
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model


model = create_model()

history = model.fit_generator(
    generator=generator(train_samples, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_samples)/BATCH_SIZE*6,
    validation_data=generator(validation_samples, batch_size=BATCH_SIZE),
    validation_steps=len(validation_samples)/BATCH_SIZE*6,
    epochs=5,
    verbose=1)

model.summary()
model.save('model.h5')

show_stats(history)


