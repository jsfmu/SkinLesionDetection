import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import datetime

# 1. Load and Explore the Dataset
print("Loading dataset...")
df = pd.read_csv('HAM10000_images/archive/HAM10000_metadata.csv')

# Display basic information
print("\nDataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualize the distribution of lesion types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='dx')
plt.title('Distribution of Lesion Types')
plt.xticks(rotation=45)
plt.savefig('lesion_distribution.png')
plt.close()

# 2. Data Preprocessing
print("\nPreprocessing data...")
# Create a binary classification (malignant vs benign)
df['malignant'] = df['dx'].map({
    'bcc': 'malignant',  # Basal cell carcinoma
    'mel': 'malignant',  # Melanoma
    'akiec': 'malignant',  # Actinic keratoses
    'bkl': 'benign',  # Benign keratosis
    'df': 'benign',   # Dermatofibroma
    'nv': 'benign',   # Melanocytic nevi
    'vasc': 'benign'  # Vascular lesions
})

# Add .jpg extension to image_id
df['image_id'] = df['image_id'] + '.jpg'

# 3. Create Data Generators with Enhanced Augmentation
print("\nSetting up data generators...")
img_height = 224
img_width = 224
batch_size = 32

# Enhanced training data augmentation with more aggressive parameters
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 4. Implement K-Fold Cross Validation
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 5. Load existing model or create new one
def load_or_create_model():
    try:
        print("Loading existing model...")
        model = load_model('skin_lesion_model.h5')
        print("Model loaded successfully!")
    except:
        print("No existing model found. Creating new model...")
        model = create_model()
    return model

def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model

# 6. Training with K-Fold Cross Validation
print("\nStarting K-Fold Cross Validation Training...")
fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_idx, val_idx in kfold.split(df):
    print(f'\nFold {fold_no}')
    
    # Split data
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory='HAM10000_images/archive/HAM10000_images_part_1',
        x_col='image_id',
        y_col='malignant',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory='HAM10000_images/archive/HAM10000_images_part_1',
        x_col='image_id',
        y_col='malignant',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Load or create model
    model = load_or_create_model()
    
    # Compile model with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f'best_model_fold_{fold_no}.h5',
            monitor='val_accuracy',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # More aggressive learning rate reduction
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model with more epochs
    history = model.fit(
        train_generator,
        epochs=100,  # Increased number of epochs
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Evaluate model
    scores = model.evaluate(validation_generator)
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    fold_no += 1

# 7. Print Results
print('\nResults per fold:')
for i in range(len(acc_per_fold)):
    print(f'Fold {i+1} - Accuracy: {acc_per_fold[i]:.4f} - Loss: {loss_per_fold[i]:.4f}')
print(f'\nAverage accuracy: {np.mean(acc_per_fold):.4f} (+/- {np.std(acc_per_fold):.4f})')

# 8. Save the best model
best_fold = np.argmax(acc_per_fold)
best_model = load_model(f'best_model_fold_{best_fold}.h5')
best_model.save('skin_lesion_model.h5')

print("\nTraining complete! Best model saved as 'skin_lesion_model.h5'")

# Function to make predictions on new images
def predict_lesion(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = best_model.predict(img_array)
    probability = prediction[0][0]
    
    if probability > 0.5:
        return "Malignant", probability
    else:
        return "Benign", 1 - probability

print("\nTraining and evaluation complete!") 