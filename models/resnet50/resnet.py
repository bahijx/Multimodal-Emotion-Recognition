import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from collections import defaultdict
from tensorflow.keras.optimizers import SGD

# Define directories for your dataset
train_dir = 'datasets/train'
test_dir = 'datasets/test'

# Function to load images and labels from a directory and balance the dataset
def load_images_and_labels(directory, exclude_emotions=None, balance=False):
    images = []
    labels = []
    class_counts = defaultdict(int)
    min_samples = float('inf')
    
    # Calculate the minimum number of samples among all classes
    for emotion_folder in os.listdir(directory):
        if exclude_emotions and emotion_folder in exclude_emotions:
            continue  # Skip this emotion if it's in the exclusion list
        emotion_path = os.path.join(directory, emotion_folder)
        if os.path.isdir(emotion_path):
            num_samples = len(os.listdir(emotion_path))
            class_counts[emotion_folder] = num_samples
            min_samples = min(min_samples, num_samples)
    
    # Sample the same number of instances for each class to create a balanced dataset if requested
    for emotion_folder, num_samples in class_counts.items():
        emotion_path = os.path.join(directory, emotion_folder)
        if os.path.isdir(emotion_path):
            if balance:  # Balance the dataset by sampling the minimum number of instances per class
                num_samples = min_samples
            for image_file in np.random.choice(os.listdir(emotion_path), num_samples, replace=False):
                image_path = os.path.join(emotion_path, image_file)
                image = load_img(image_path, target_size=(224, 224))
                image_array = img_to_array(image)
                images.append(image_array)
                labels.append(emotion_folder)
    
    return np.array(images), np.array(labels)

# Load balanced training and testing images and labels, excluding 'contempt' and 'disgust'
X_train, y_train = load_images_and_labels(train_dir, exclude_emotions=['contempt', 'disgust'], balance=True)
X_test, y_test = load_images_and_labels(test_dir, exclude_emotions=['contempt', 'disgust'], balance=False)

# Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train_encoded))
y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

# Load the ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for your specific task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Adjust the number of output classes as needed

# Combine the base model with custom top layers
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune the pre-trained ResNet model
for layer in model.layers[:-10]:  # Fine-tune the last 10 layers
    layer.trainable = True

# Define the SGD optimizer with custom parameters
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

# Compile the model with the custom SGD optimizer
model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Shuffle the training data
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train_shuffled = X_train[indices]
y_train_shuffled = y_train_categorical[indices]

# Train the model
history = model.fit(
    X_train_shuffled, y_train_shuffled,
    epochs=10,
    validation_data=(X_test, y_test_categorical)
)

model.save('models/resnet50/resnet50_emotion_classifier.h5')

# Calculate the average validation accuracy across all folds
average_acc = np.mean(history.history['val_accuracy'])
print(f'Average Validation Accuracy: {average_acc}')
