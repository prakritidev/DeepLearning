import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# batch size defines number of samples whic are going to propogate in the network
batch_size = 32
#target classes
num_classes = 10
#e 1 epoch = (forward + backword) propogation.
epochs = 300 #
data_augmentation = True

# data shuffling and splitting into traininf and testing sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'training samples')
print(x_test.shape[0], 'testing samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('------------y_test------------')
print(x_train.shape[1:])

# creating model 
# The Sequential model is a linear stack of layers. 
# you can add layer ain a sequential layer just like a stack.
# Layers Stacked in sequential model for CIFAR 10
'''
convolution layer
		|
		v
    Activation
		|
		v
convolution layer
		|
		v
	Activation(relu)
		|
		v
    MaxPooling
    	|
    	v
     Dropout
     	|
     	v

convolution layer
		|
		v
    Activation(relu)
		|
		v
convolution layer
		|
		v
	Activation(relu)
		|
		v
    MaxPooling
    	|
    	v
     Dropout
     	|
     	v

	 Flatten
		|
		v
	  Dense
		|
		v
    Activation(relu)
    	|
    	v
     Dropout
     	|
     	v
      Dense
      	|
      	v
    Activation(softmax)
'''

# implemnting above stack into code.
model = Sequential()
# adding layers to the sequential model 
# .add() will push the layer into the stack(sequential model) 
# filter -> 32
# strides ->(3,3)
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
# reulu is an activation function faster than sigmoid and reduce the liklehod of vanishing gradient 
model.add(Activation('relu'))

# we donr have to define parameters again.
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# Maxpooling is done for downsampling the input and reducing its dimensionality
# and also help over-fitting by providing an abstracted form of the representation
model.add(MaxPooling2D(pool_size=(2, 2)))
# drop put is a regularization technique where randomly selected neurons are ignored during training
# here 25% connection will drop
model.add(Dropout(0.25))

# Depth is changed into 64
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


# conver matrix into a vector by streching it.
model.add(Flatten())
# Dense layer is use to collect all the info 
model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))
# number of target classes
model.add(Dense(num_classes))
# softmax convert probabilty into binary number.
model.add(Activation('softmax'))
# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
