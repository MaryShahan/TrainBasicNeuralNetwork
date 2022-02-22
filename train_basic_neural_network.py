import numpy as np
import pandas as pd

bias = 0.5
l_rate = 0.1
epochs = 10
epoch_loss = []

rg = np.random.default_rng() #random generator for random floating point numbers

def generate_data(n_features, n_values):
    features = rg.random((n_features, n_values))
    weights = rg.random((1, n_values))[0] # [0] cause we need it one-dimensional [] not 2D [[]]
    targets = np.random.choice([0,1], n_features)
    print("features:", features)
    print("weights:", weights)
    print("targets:", targets)
    print("*****************************************")

    data = pd.DataFrame(features, columns=["x0", "x1", "x2"])
    data["targets"] = targets
    print(data)
    return data, weights

data, weights = generate_data(4,3)

def get_weighted_sum(feature, weights, bias):
    return np.dot(feature, weights) + bias

def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

def cross_entropy(target, prediction):
    return -(target*np.log10(prediction) + (1-target)*(np.log10(1-prediction)))

def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x,w in zip(feature, weights):
        new_w = w +l_rate*(target-prediction)*x
        new_weights.append(new_w)
    return  new_weights

def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate*(target-prediction)

def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        individual_loss = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            #print("*****************************************")
            #print(feature)
            w_sum = get_weighted_sum(feature, weights, bias)
            #print("*****************************************")
            #print("w_sum:", w_sum)
            #print("*****************************************")
            prediction = sigmoid(w_sum)
            #print("prediction:", prediction)
            #print("*****************************************")
            loss = cross_entropy(target, prediction)
            #print("loss:" , loss)
            individual_loss.append(loss)
            #Gradiant Descent
            #print("old value:")
            #print(weights, bias)
            weights = update_weights(weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
            #print("new value:")
            #print(weights, bias)
        average_loss = sum(individual_loss)/len(individual_loss)
        epoch_loss.append(average_loss)
        print("*****************************************")
        print("epoch", e)
        print(average_loss)

train_model(data, weights, bias, l_rate, epochs)


#plot the average loss
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("Training_loss.pdf")




