# MTGCardColorPredictor-ML

A model that predicts the color of the card based on the first 50 words of the card text with keras. It achieved around an 85% accuracey rating during testing, which is great for how simple the model is. The model is a simple DNN with 3 deep layers and dropout, binary cross entropy for loss (because multilabel classification) and Adam optimizer.

To create the model, run ModelCreation.py. To test Kaldheim Cards (Most recent set and model did not train on the set), run ModelEvaluate.py. To test specific text, run ModelEvaluateTest.py, changing the text in the file.
