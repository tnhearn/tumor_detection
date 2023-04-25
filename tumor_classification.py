import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import pandas as pd
import seaborn as sns

# Define the path to the dataset
img_path = '/Users/thear/Documents/Code/tumor_detection/brain_tumor_dataset/'

# Create a list of all the image filenames
all_images = []
for folder in ['yes', 'no']:
    folder_path = os.path.join(img_path, folder)
    for filename in os.listdir(folder_path):
        
         all_images.append(os.path.join(folder_path, filename))

# Create a list of corresponding labels (0 for 'no', 1 for 'yes')
labels = ['Tumor' if 'Y' in filename else 'Healthy' for filename in all_images]

all_images = pd.DataFrame({'filename': all_images, 'status': labels})

# Split the dataset into train, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_images, 
    labels, 
    test_size = 0.05, 
    random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val, 
    test_size = 0.25, 
    random_state = 42)



# Create a 2x4 subplot
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 6))
img_to_plot  = [0, 1, 2, 3, len(labels)-1, len(labels)-2, len(labels)-3, len(labels)-4]

# Loop over image paths and plot each image on the subplot
for i, ax in enumerate(axs.flat):
    # Load image from file
    img = imread(all_images.iloc[img_to_plot[i],0])

    # Display image on subplot
    ax.imshow(img)

    # Remove ticks and labels from subplot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if i < 4:
        ax.set_title('Tumor (1)')
    else:
        ax.set_title('Healthy (0)')


batch_size = 16

# Create an instance of ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    #Randomly increase or decrease the size of the image by up to 10%
    zoom_range = 0.1, 
    #Randomly rotate the image between -25,25 degrees
    rotation_range = 25, 
    #Shift the image along its width by up to +/- 5%
    width_shift_range = 0.05, 
    #Shift the image along its height by up to +/- 5%
    height_shift_range = 0.05,
    )

# Create a generator for the training set
train_generator = train_datagen.flow_from_dataframe(
    dataframe = X_train,
    directory = None,
    x_col = 'filename',
    y_col = 'status',
    target_size = (256, 256),
    batch_size = batch_size,
    class_mode = 'binary',
    color_mode = 'grayscale')

# Create a generator for the validation set
val_datagen = ImageDataGenerator(rescale = 1./255)

val_generator = val_datagen.flow_from_dataframe(
    dataframe = X_train_val,
    directory = None,
    x_col = 'filename',
    y_col = 'status',
    target_size = (256, 256),
    batch_size = batch_size,
    class_mode = 'binary',
    color_mode = 'grayscale')

def design_model():
    #Build model
    model = Sequential()
    
    model.add(InputLayer(input_shape = (256, 256, 1)))
    
    model.add(Conv2D(256, 3, strides = 3, activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, 3, strides = 3, activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, 3, strides = 3, activation = 'relu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(
        optimizer = Adam(learning_rate = 0.001),
        loss = BinaryCrossentropy(), 
        metrics = [BinaryAccuracy()])
    
    return model


# early stopping implementation
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose = 1, 
                   patience = 20)

# Apply the model
model = design_model()

print(model.summary())

history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.samples/batch_size,
    epochs = 100,
    validation_data = val_generator,
    validation_steps = val_generator.samples/batch_size,
    callbacks = [es]
    )

# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['binary_accuracy'])
ax1.plot(history.history['val_binary_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(['Train', 'Validation'], loc='upper left')

# plotting auc and validation auc over epochs
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(['train', 'validation'], loc='upper left')

plt.show()


test_steps_per_epoch = np.math.ceil(val_generator.samples / val_generator.batch_size)

predictions = model.predict(val_generator, 
                            steps = None)

test_steps_per_epoch = np.math.ceil(val_generator.samples / val_generator.batch_size)

predicted_classes = (predictions > 0.5).astype(int)

true_classes = val_generator.classes

class_labels = list(val_generator.class_indices.keys())

report = classification_report(true_classes, 
                               predicted_classes, 
                               target_names = class_labels)

print(report)   
 
cm=confusion_matrix(true_classes,predicted_classes)

# Plot confusion matrix
fig, ax3 = plt.subplots(figsize=(5, 5))
heatmap = sns.heatmap(
    cm, 
    fmt = 'g', 
    cmap = 'mako_r', 
    annot = True, 
    ax = ax3)
ax3.set_xlabel('Predicted class')
ax3.set_ylabel('True class')
ax3.set_title('Confusion Matrix')
ax3.xaxis.set_ticklabels(class_labels)
ax3.yaxis.set_ticklabels(class_labels)






