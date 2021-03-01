import numpy as np
import Glove
import json
import tensorflow as tf
from tensorflow import keras
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = keras.models.load_model("CardColorCategorizer.h5")

allCardDataStr = ""
with open('KaldheimCardsInput.txt', 'r', encoding="utf8") as myfile:
    allCardDataStr = json.load(myfile)
glove_model = Glove.loadGloveModel()
inputs = []
outputs = []
allnames = []
for card in allCardDataStr:
    keywordVector = []
    finalKeywordVector = []
    allnames.append(card["name"])
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

y = model.predict(inputs)
y_mod = []

for rry in y:
    y_rry = []
    for val in rry:
        if val >= 0.9:
            val = 1
        else:
            val = 0
        y_rry.append(val)
    y_mod.append(y_rry)
y_mod = np.array(y_mod)
right = 0
total = 0
for idx in range(len(outputs)):
    pred = y_mod[idx]
    output = outputs[idx]
    for idx_rry in range(len(pred)):
        total += 1
        if pred[idx_rry] == output[idx_rry]:
            right += 1
        else:
            print(allnames[idx])
x = 5