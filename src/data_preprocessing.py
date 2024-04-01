from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize ImageDataGenerators for training and testing
# Rescale the images to normalize pixel values from 0-255 to 0-1
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test data
train_dir = 'datasets/train'
test_dir = 'datasets/test'

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(48, 48),  # All images will be resized to 48x48
        batch_size=32,
        color_mode='grayscale',  # Convert images to grayscale
        class_mode='categorical')  # Since we use categorical_crossentropy loss, we need categorical labels

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical')
