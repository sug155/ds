import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Preprocessing & Loading Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    'Animals',                # Path to your dataset folder
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'Animals',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')  # 3 classes: cats, dogs, snakes
])

# 3. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# 5. Evaluate the model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.2f}")

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Load the saved model (if you've saved it before)
# model = tf.keras.models.load_model('your_model_path.h5')  # Uncomment if saved
# If the model is still in memory, you can use it directly without loading

# 2. Define class names (in the same order as the folders)
class_names = ['cats', 'dogs', 'snakes']

# 3. Load and preprocess the image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # same size used during training
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # 4. Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

# Example usage 
predict_image('image.jpg')  # replace with your image path