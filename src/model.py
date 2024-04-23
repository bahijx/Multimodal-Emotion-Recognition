import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_resnet_model(num_classes):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add new layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
    x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
    predictions = Dense(num_classes, activation='softmax')(x)  # Add a logistic layer for `num_classes` classes
    
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
num_classes = 7  # Adjust to the number of classes in your dataset
model = build_resnet_model(num_classes)

# Summary of the model
model.summary()
