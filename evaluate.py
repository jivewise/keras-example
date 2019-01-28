"""Neural network evaluation code

This script evaluates a neural network model, makes predictions, and can be used for debugging
a model.

    * print_layer - prints the inputs and outputs of a Keras layer
    * evaluate - predicts the value of test data using model, and plots the value against the expected result
    * evaluate_binary - predicts the value of test data, and prints out confusion matrix
    * run - uses above functions to evaluate a model and output debugging info

"""
import numpy as np
from keras import backend as K
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, f1_score
from functools import reduce

def print_layer(input, layer):
    print("************************************")
    print("LAYER INFO")
    print("************************************")
    print(input)
    layer_function = K.function([layer.input], [layer.output])
    output = layer_function(input)
    print(output)
    return output

def evaluate(model, data):
    print("************************************")
    print("Prediction")
    print("************************************")

    X_test = data.X_test
    y_test = data.y_test

    y_pred = model.predict(X_test)
    print(y_pred)
    y_pred_scaled = data.unscale_y(y_pred)
    y_scaled = data.unscale_y(y_test)

    pyplot.plot(y_pred_scaled, label='Prediction')
    pyplot.plot(y_scaled, label='Actual')
    pyplot.legend()
    pyplot.show()

    print("************************************")
    print("Layer outputs")
    print("************************************")
    reduce(print_layer, model.layers, [data.X_train])

    return y_pred_scaled

def evaluate_binary(model, data):
    X = data.X_test
    y = data.y_test

    print("************************************")
    print("Prediction")
    print("************************************")

    y_pred = model.predict(X)
    y_scaled = np.interp(y_pred, (y_pred.min(), y_pred.max()), (0, +1))

    print("************************************")
    print("F1 Score")
    print("************************************")
    y_pred_5 = (np.squeeze(y_scaled) > 0.5)
    cm = confusion_matrix(y, y_pred_5)
    print("CM:", cm)
    f1 = f1_score(y, y_pred_5)
    print("F1 Score", f1)

    return y_pred_5

def run(model, data, predict_value=False):
    """Runs evaluation on the model using data
    Parameters
    ----------
    model : keras.engine.sequential.Sequential
        Model for evaluation
    data : DataSet
        DataSet object filled with scaled and split data ready for consumption
    predict_values : Boolean
        Set to true if you want to predict the value of the stock or false if you want to categorize it
    epochs : Int
        How many epochs you want the network to run
    Returns
    -------
    List
        Predictions for X_test in data, comparable to y_test
    """
    y_pred = evaluate(model, data) if predict_value else evaluate_binary(model, data)
    return y_pred
