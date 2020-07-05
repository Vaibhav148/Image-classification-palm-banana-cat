import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np
from keras.preprocessing import image
import os

# Data Preprocessing

# Preprocessing the Training set

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 40,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 10,
                                                 class_mode = 'categorical',
                                                 shuffle = True)

# Preprocessing the Test set

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 3,
                                            class_mode = 'categorical')


#Building the CNN

#Initialising the CNN
cnn = tf.keras.models.Sequential()


#Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


#Pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


#Flattening
cnn.add(tf.keras.layers.Flatten())


#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


#Output Layer
cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

cnn.summary()


# Training the CNN

#Compiling the CNN

cnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Training the CNN on the Training set and evaluating it on the Test set

history = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
pyplot.legend()
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['acc'], color='blue', label='train')
pyplot.plot(history.history['val_acc'], color='orange', label='test')
pyplot.legend()
pyplot.tight_layout(pad=1.0)





#Making a single prediction

test_image = image.load_img('dataset/single_prediction/cat.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict_classes(test_image)

if result[0]==0:
    print("Banana")
elif result[0]==1:
    print("Cat")
else:
    print("Palm")

#Making the prediction on banana test set
images = []
img_folder = os.path.join('dataset/test_set/banana)
img_files = os.listdir(img_folder)
img_files = [os.path.join(img_folder,f) for f in img_files]
for img in img_files:
    img = image.load_img(img, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)

images = np.vstack(images)
classes = cnn.predict_classes(images, batch_size=10)
print(classes)
#class 0 - banana , class 1 – cat , class 2 - palm 


# Saving the weights 
model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
cnn.save_weights("model.h5")
print("Saved model to disk")



