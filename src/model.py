import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.regularizers import l2

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10, l2_reg=0.0001):
    """
    Builds and compiles a Convolutional Neural Network (CNN) for character recognition.
    
    Args:
        input_shape (tuple): Shape of the input images, defaults to (28, 28, 1).
        num_classes (int): Number of target classes, e.g., 10 for MNIST, 47 for EMNIST Balanced.
        l2_reg (float): L2 regularization factor.
        
    Returns:
        model (Sequential): Compiled Keras Sequential model.
    """
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First convolutional block
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flattening and fully connected layers
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(0.5),
        
        # Output layer with softmax activation
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    # Print the model summary for verification
    model = build_cnn_model(num_classes=47)
    model.summary()
