import numpy as np
import Glove
import json
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = keras.models.load_model("CardColorCategorizer.h5")
glove_model = Glove.loadGloveModel()
text = []
newText = "flash enchant creature or vehicle enchanted creature gets minus three minus zero"
text.append(newText.split())

inputs = []
outputs = []
for input in text:
    keywordVector = []
    finalKeywordVector = []
    for word in input:
        if word in glove_model:
            keywordVector.append(glove_model[word])
    for i in range(50):
        if len(keywordVector) > i:
            finalKeywordVector.extend(keywordVector[i])
        else:
            finalKeywordVector.extend(np.zeros(50).tolist())
    finalKeywordVector = np.array(finalKeywordVector)
    inputs.append(finalKeywordVector)
inputs = np.array(inputs)
outputs = np.array(outputs)

y = model.predict(inputs)
y_mod = []