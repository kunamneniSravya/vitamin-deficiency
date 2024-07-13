import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Part 1: Building the CNN model

# Initialize the CNN model
np.random.seed(1337)
classifier = Sequential()

# Add convolutional layers
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(16, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(8, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layers
classifier.add(Flatten())

# Add fully connected layers
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=6, activation='softmax'))  # Update units to 6 for 6 classes

# Compile the model
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2: Fitting the dataset

# Preprocess and augment the training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Preprocess the testing data
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training data
training_set = train_datagen.flow_from_directory(
    r'C:\Users\Jeevan\OneDrive\minor\Vitamin\train',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Load the testing data
test_set = test_datagen.flow_from_directory(
    r'C:\Users\Jeevan\OneDrive\minor\Vitamin\train',
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Calculate steps per epoch and validation steps
steps_per_epoch = len(training_set)
validation_steps = len(test_set)

# Fit the model to the training data
classifier.fit_generator(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=78,
    validation_data=test_set,
    validation_steps=validation_steps
)

# Part 3: Making predictions

# Make predictions using the trained model on the test set
predictions = classifier.predict_generator(test_set)

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Get the true classes
true_classes = test_set.classes

# Get the class labels
class_labels = list(test_set.class_indices.keys())

# Print some predictions
print("Predicted classes:", predicted_classes[:10])
print("True classes:", true_classes[:10])
print("Class labels:", class_labels)

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print("Accuracy:", accuracy)

# Save the trained model weights
classifier.save_weights('keras_vitamin_trained_model_weights.h5')
print('Saved trained model as "keras_vitamin_trained_model_weights.h5"')
