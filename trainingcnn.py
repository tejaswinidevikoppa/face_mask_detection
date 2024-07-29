import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Loading the saved numpy arrays
data = np.load('data.npy')
target = np.load('target.npy')

# Define the model
model = Sequential()

# First convolutional layer
model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to stack the output convolutions from second convolutional layer
model.add(Flatten())
model.add(Dropout(0.5))

# Dense layer
model.add(Dense(50, activation='relu'))

# Final layer with two outputs for two categories
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the data into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

# Checkpoint to save the best model
checkpoint = ModelCheckpoint('model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

# Train the model
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)
