import numpy as np
import Glove
import json
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random

allCardDataStr = ""
with open('mtgCardColorInput.txt', 'r', encoding="utf8") as myfile:
    allCardDataStr = json.load(myfile)
glove_model = Glove.loadGloveModel()
inputs = []
outputs = []
for card in allCardDataStr:
    keywordVector = []
    finalKeywordVector = []
    for word in card["words"]:
        if word in glove_model:
            keywordVector.append(glove_model[word])
    for i in range(50):
        if len(keywordVector) > i:
            finalKeywordVector.extend(keywordVector[i])
        else:
            finalKeywordVector.extend(np.zeros(50).tolist())
    finalKeywordVector = np.array(finalKeywordVector)
    inputs.append(finalKeywordVector)
    outputs.append(np.array(card["colors"]))
inputs = np.array(inputs)
outputs = np.array(outputs)
EightyPerc = int(len(inputs) * .8)
TwentyPerc = int(len(inputs) * .2)
#x_train = tf.data.Dataset.from_tensor_slices(inputs[:EightyPerc])
#y_train = tf.data.Dataset.from_tensor_slices(outputs[:EightyPerc])
train_dataset = tf.data.Dataset.from_tensor_slices((inputs[:EightyPerc], outputs[:EightyPerc])).shuffle(500).repeat(1).batch(54)
#x_test = tf.data.Dataset.from_tensor_slices(inputs[EightyPerc:])
#y_test = tf.data.Dataset.from_tensor_slices(outputs[EightyPerc:])
test_dataset = tf.data.Dataset.from_tensor_slices((inputs[EightyPerc:], outputs[EightyPerc:])).shuffle(500).repeat(1).batch(54)

input_neurons = len(inputs[0])
output_neurons = len(outputs[0])

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_neurons))
model.add(keras.layers.Dense(1500, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(750, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(500, activation=keras.activations.relu))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Dense(output_neurons, activation=keras.activations.sigmoid))
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=keras.metrics.binary_accuracy)
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.save("CardColorCategorizer.h5")

