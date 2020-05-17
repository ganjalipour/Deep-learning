import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist= tf.keras.datasets.mnist

(x_train,y_train), (x_test,y_test) =mnist.load_data()

##### Normalize input features
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)



##### Define NN Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

##### Train
model.fit(x_train,y_train,epochs=2)

##### Save Model

model.save('ganjNNModel.model')

##### Evaluate NNModel with test data
ganj_model= tf.keras.models.load_model('ganjNNModel.model')

val_loss, val_accuracy = ganj_model.evaluate(x_test,y_test)

##### Prediction
predictions = model.predict([x_test])
print("predicted Class is :")
print (np.argmax(predictions[5]))



plt.imshow(x_train[5])
plt.show()
