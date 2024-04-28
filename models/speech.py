import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, MaxPooling1D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def build_emotion_cnn(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = Conv1D(filters=256, kernel_size=5, strides=1, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)

    # Second convolutional block
    x = Conv1D(filters=128, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Third convolutional block
    x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Output layer
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Load dataset (replace 'your_dataset.csv' with your dataset path)
data = pd.read_csv('datasets/complete.csv')
data.fillna(0, inplace=True)

# Remove rows with "fearful" emotion
data = data[data['Emotion'] != 'Fearful']
data = data[data['Emotion'] != 'Surprised']



# Remove rows with "fearful" and "surprised" emotions
data = data[(data['Emotion'] != 'Fearful') & (data['Emotion'] != 'Surprised')]

# Extract values from comma-separated strings in 'MFCCs12' column and put them into arrays
data['MFCCs12'] = data['MFCCs12'].apply(lambda x: np.array([float(value) for value in x.split(',')]))

# Extract values from 'MFCCs12' into variables mfccs1 to mfccs12
for i in range(12):
    data[f'mfccs{i+1}'] = data['MFCCs12'].apply(lambda x: x[i])


# Extract values from comma-separated strings in 'MFCCs12' column and put them into arrays
data['MFCCs18'] = data['MFCCs18'].apply(lambda x: np.array([float(value) for value in x.split(',')]))

# Extract values from 'MFCCs12' into variables mfccs1 to mfccs12
for i in range(18):
    data[f'mfccs{i+1}'] = data['MFCCs18'].apply(lambda x: x[i])
    
# Extract values from comma-separated strings in '128logmelspectogram' column and put them into arrays
data['128logmelspectogram'] = data['128logmelspectogram'].apply(lambda x: np.array([float(value) for value in x.split(',')]))

# Extract values from '128logmelspectogram' into variables logmel1 to logmel128
for i in range(128):
    data[f'logmel{i+1}'] = data['128logmelspectogram'].apply(lambda x: x[i])
    
  # Extract values from comma-separated strings in 'Energies' column and put them into arrays
data['Energies'] = data['Energies'].apply(lambda x: np.array([float(value) for value in x.split(',')]))

# Extract values from 'Energies' into variables energy1 to energy4
for i in range(4):
    data[f'energy{i+1}'] = data['Energies'].apply(lambda x: x[i])
  
    
    
    
# Define selected features including mfccs1 to mfccs12
svf1_selected_features =  [f'mfccs{i+1}' for i in range(12)]+ ['F1_mean', 'F2_mean', 'F3_mean', 'LTAS_min', 'LTAS_max', 'LTAS_mean', 
                     'LTAS_std', 'LTAS_range', 'LTAS_slope']
prosodic_selected_features = ['Min_pitch', 'Max_pitch', 'Mean_pitch', 'SD_pitch',
                     'Min_intensity', 'Max_intensity', 'Mean_intensity', 'SD_intensity',
                     'Local_shimmer', 'Local_dB_shimmer', 'RAP_jitter', 'PPQ5_jitter',
                     'Local_jitter', 'Local_absolute_jitter',
                     'Pitch_range', 'Intensity_range',
                     'APQ3_shimmer', 'APQ5_shimmer']
svf2_selected_features=[f'logmel{i+1}' for i in range(128)] + [f'mfccs{i+1}' for i in range(18)]
wavelet_selected_features= [f'energy{i+1}' for i in range(4)]+['Wavelet_entropy']

features=svf2_selected_features+svf1_selected_features+prosodic_selected_features


X = data[features]  # SFV1&SVF2&PFV
y = data['Emotion']  # Labels

# Determine the minimum number of samples among all emotion classes
min_samples = min(data['Emotion'].value_counts())

# Sample the same number of instances for each class to create a balanced dataset
balanced_data = pd.concat([data[data['Emotion'] == emotion].sample(min_samples, replace=True) for emotion in np.unique(y)])

# Reassign X and y with balanced data
X = balanced_data[features]
y = balanced_data['Emotion']

# Print balanced dataset
print("Balanced dataset shape:", balanced_data.shape)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Print data shape after normalization
print("Data shape after normalization:", X_normalized.shape)

# Reshape input data
input_length = X.shape[1]
X_normalized = X_normalized.reshape(-1, input_length, 1)

# Print data shape after reshaping
print("Data shape after reshaping:", X_normalized.shape)

# Initialize cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store accuracy for each fold
acc_per_fold = []

# Perform 10-fold cross-validation
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_normalized, y), 1):
    print(f'Fold {fold_idx}:')

    # Split data into train and validation sets for this fold
    X_train, X_val = X_normalized[train_idx], X_normalized[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Get number of classes
    num_classes = len(np.unique(y_train))

    # Build the model
    input_shape = (X_train.shape[1], 1)
    model = build_emotion_cnn(input_shape, num_classes)

    # Compile model with SGD optimizer
    optimizer = SGD(learning_rate=0.001, momentum=0.8)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    print("Training...")
    history = model.fit(X_train, y_train, epochs=80, batch_size=16, 
                        validation_data=(X_val, y_val), verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    # Evaluate model on validation set
    scores = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation Accuracy: {scores[1]}')

    # Store accuracy for this fold
    acc_per_fold.append(scores[1])

# Print average validation accuracy across all folds
average_acc = np.mean(acc_per_fold)
print(f'Average Validation Accuracy: {average_acc}')