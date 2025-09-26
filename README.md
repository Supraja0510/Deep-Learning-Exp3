# Deep-Learning-Exp3

**DL-Convolutional Deep Neural Network for Image Classification**

**AIM**

To develop a convolutional neural network (CNN) classification model for the given dataset.

**THEORY**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

**Neural Network Model**

<img width="815" height="456" alt="image" src="https://github.com/user-attachments/assets/3152d664-c5f5-43fa-8183-54f3975a1fbf" />

**DESIGN STEPS**

STEP 1: Preprocess the MNIST dataset by scaling the pixel values to the range [0, 1] and converting labels to one-hot encoded format.

STEP 2: Build a convolutional neural network (CNN) model with specified architecture using TensorFlow Keras.

STEP 3: Compile the model with categorical cross-entropy loss function and the Adam optimizer.

STEP 4: Train the compiled model on the preprocessed training data for 5 epochs with a batch size of 64.

STEP 5: Evaluate the trained model's performance on the test set by plotting training/validation metrics and generating a confusion matrix and classification report. Additionally, make predictions on sample images to demonstrate model inference.

**PROGRAM**

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('7.png')

type(img)

img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```


Name: SUPRAJA B

Register Number: 2305002026

class CNNClassifier(nn.Module):
   
    def __init__(self, input_size):
       
        super(CNNClassifier, self).__init__()
       
        #Include your code here

    def forward(self, x):
       
        #Include your code here

       ** # Initialize the Model, Loss Function, and Optimizer**
  
       model =
  
       criterion =
  
       optimizer =

       def train_model(model, train_loadr, num_epochs=10):
   
      #Include your code here

**OUTPUT**

**Training Loss per Epoch**

<img width="815" height="152" alt="image" src="https://github.com/user-attachments/assets/b22d1fc9-4c8a-4b96-a673-ee231b088620" />


**Confusion Matrix**

<img width="816" height="441" alt="image" src="https://github.com/user-attachments/assets/452c01a9-8cee-4b67-a88f-835cbf8e757e" />


**Classification Report**

<img width="672" height="283" alt="image" src="https://github.com/user-attachments/assets/c6da6baa-3617-4c5c-a0ad-ef10fe401d5a" />

**New Sample Data Prediction**

<img width="760" height="683" alt="image" src="https://github.com/user-attachments/assets/07f5fe74-c385-4b2c-930f-4e2b7423cdb7" />

<img width="729" height="696" alt="image" src="https://github.com/user-attachments/assets/321568b7-feae-46cf-9e8a-1cc4fc790b9f" />

<img width="816" height="574" alt="image" src="https://github.com/user-attachments/assets/972ae2fd-6541-4d76-a5f9-eda62d64c497" />

<img width="676" height="694" alt="image" src="https://github.com/user-attachments/assets/395e3a9e-0f24-447e-b00a-468bb1f7dfde" />


**RESULT**

Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed successfully.
