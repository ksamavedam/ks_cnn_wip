import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Generate synthetic data for demonstration
# Replace this with your actual data loading and preprocessing
num_grids = 16
num_samples = 100
num_sources = 3
grid_size = 4

# Generate synthetic data (random RSSI values)
X = np.random.rand(num_grids, num_samples, num_sources)
y = np.arange(num_grids)  # Assuming the labels are 0 to 15 for the 16 grids

# Step 4: Model Architecture
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(num_samples, num_sources)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_grids, activation='softmax')  # Assuming you have 16 grids for classification
])

# Step 6: Compile the Model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Step 7: Training
model.fit(X, y, epochs=10, batch_size=32)

# Step 8: Evaluation (Optional, if you have a separate test set)
# test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Step 10: Prediction
# Assuming X_new_data is your new data in the shape (num_samples, num_sources)
predictions = model.predict(np.expand_dims(X_new_data, axis=0))  # Expand dimensions to match model input shape
predicted_grid = np.argmax(predictions)
print(f"Predicted Grid: {predicted_grid}")
