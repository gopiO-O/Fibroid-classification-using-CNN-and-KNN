import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

train_dir = r'C:\Users\91709\Downloads\Fibroid datasets\train'
test_dir = r'C:\Users\91709\Downloads\Fibroid datasets\test'

img_height, img_width = 128, 128
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(train_generator, epochs=10, validation_data=test_generator)

cnn_model.save('cnn_fibroid_model.h5')

cnn_preds = cnn_model.predict(test_generator)
cnn_preds = np.round(cnn_preds).astype(int)

cnn_cm = confusion_matrix(test_generator.classes, cnn_preds)
print("CNN Confusion Matrix:\n", cnn_cm)
print("CNN Classification Report:\n", classification_report(test_generator.classes, cnn_preds))

def extract_features(generator, model):
    features = model.predict(generator)
    return features

train_features = extract_features(train_generator, cnn_model)
test_features = extract_features(test_generator, cnn_model)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(train_features, train_generator.classes)

knn_preds = knn_model.predict(test_features)

knn_cm = confusion_matrix(test_generator.classes, knn_preds)
print("KNN Confusion Matrix:\n", knn_cm)
print("KNN Classification Report:\n", classification_report(test_generator.classes, knn_preds))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(cnn_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('CNN Confusion Matrix')
axes[1].imshow(knn_cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].set_title('KNN Confusion Matrix')

for ax in axes:
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Non-Fibroid', 'Fibroid'])
    ax.set_yticklabels(['Non-Fibroid', 'Fibroid'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.show()
