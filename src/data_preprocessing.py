from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

# Initialize ImageDataGenerators for training and testing
# Only rescale the images without any other augmentation
train_datagen_resnet = ImageDataGenerator(rescale=1./255, preprocessing_function=resnet_preprocess_input)
test_datagen_resnet = ImageDataGenerator(rescale=1./255, preprocessing_function=resnet_preprocess_input)

train_datagen_vgg19 = ImageDataGenerator(rescale=1./255, preprocessing_function=vgg19_preprocess_input)
test_datagen_vgg19 = ImageDataGenerator(rescale=1./255, preprocessing_function=vgg19_preprocess_input)

train_dir = 'datasets/train'
test_dir = 'datasets/test'

# Preprocess the data specifically for ResNet
train_generator_resnet = train_datagen_resnet.flow_from_directory(
        train_dir,  
        target_size=(224, 224),  # Resize images to 224x224 pixels
        batch_size=32,
        color_mode='rgb',  # ResNet and VGG19 require RGB images
        class_mode='categorical')  # No need to specify preprocessing function here

test_generator_resnet = test_datagen_resnet.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Resize images to 224x224 pixels
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical')  # No need to specify preprocessing function here

# Preprocess the data specifically for VGG19
train_generator_vgg19 = train_datagen_vgg19.flow_from_directory(
        train_dir,    
        target_size=(224, 224),  # Resize images to 224x224 pixels
        batch_size=32,
        color_mode='rgb',  # ResNet and VGG19 require RGB images
        class_mode='categorical')  # No need to specify preprocessing function here

test_generator_vgg19 = test_datagen_vgg19.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Resize images to 224x224 pixels
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical')  # No need to specify preprocessing function here