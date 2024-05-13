# weather prediction model 
# about the data set: we have a csv file that has a column for dates, precipitation, temp_max, temp_min, wind, and weather
# some of the data for the dates column is missing, it has a '####' instead of a number. We probably won't need this data anyways to predict the weather 
# for the weather column we have: drizzle, rain, snow, sun, and fog ( 5 options )

# I am not going to use all of it for training, because we can not use our test data for training. If we did, we would get an unbalanced evaluation metric
# I want to split my data 80-20, 80% will be my training set and 20% will be my test set. 

# Supervised learning : uses a training set to teach models to yield a desired output. There are two types of supervised learning but I will be using 
# classification for the weather prediction because we want to predict a categorical value () so this is a multiclass problem : ) 
# 

import torch
import wandb
#%%
# loading in the csv data 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



file = "./seattle-weather.csv" # forward slash / 
weather = pd.read_csv(file, header=0)
# make sure that the file is being read correctly 

#print(weather.head())

# get the columns 
weather.columns = [
    "date",
    "precipitation",
    "temp_max",
    "temp_min",
    "wind",
    "weather"
]


weather = weather.drop("date", axis = 1 )
#print(independent)
# have to replace the weather names with actual values so that they can be converted later in the code
weather['weather'] = weather['weather'].replace('drizzle', 0.0)
weather['weather'] = weather['weather'].replace('rain', 1.0)
weather['weather'] = weather['weather'].replace('snow', 2.0)
weather['weather'] = weather['weather'].replace('sun', 3.0)
weather['weather'] = weather['weather'].replace('fog', 4.0)
#print(independent)

#print(weather)
# the rest of the columns are going to be our dependent variables aka our labels for the neural network 

# not sure if we want to drop date from our data

dependent = weather.drop("weather", axis = 1)
independent = weather['weather'] # outcome (y)


# our model
class NeuralNetwork(nn.Module):
    # input layer  ( 4 features of our weather prediction ) --> 
    # Hidden Layer1 ( number of neurons ) -->
    # Hidden layer two (n) --> 
    # output ( the weather, 5 options )
    def __init__(self, in_features = 4, h1 = 8, h2 =9, out_features = 5): # h1 and h2 are just randomly picked numbers 
        super().__init__() # instantiate the nn.module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)

        return x
    
torch.manual_seed(32)
model = NeuralNetwork() 

#split the data into training and testing sets. so we want 80% of the data set to be training and 20% to be testing 
# convert into numpy arrays
dependent = dependent.values
independent = independent.values


# train test split
dependent_train, dependent_test, independent_train, independent_test = train_test_split(dependent, independent, test_size = 0.2, train_size= 0.8, random_state=32)

# convert the dependent values/features into float tensors 
dependent_train = torch.FloatTensor(dependent_train)
dependent_test = torch.FloatTensor(dependent_test)

# convert independent labels to tensors long 
independent_train = torch.LongTensor(independent_train)
independent_test = torch.LongTensor(independent_test)

# now we set the criterion of the model to measure error (how far off the predictions are)
criterion = nn.CrossEntropyLoss() # since we have more than one feature and output 
# choose optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# now we train our model
# epoch ( one run thru all the training data in our network <3 purr )
epochs = 100
losses = []

for i in range(epochs):
    # go forward and get a prediction 
    independent_pred = model.forward(dependent_train) # get predicted results 

    # measure the loss / error ( may be high at first )
    loss = criterion(independent_pred, independent_train) # predicted values vs the independent training values 

    # keep track of our losses
    losses.append(loss.detach().numpy())

    # print every 10 epochs 
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    # do some back propagation : take the error rate of forward propagation and feed it back through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# evaluate the model on test data set ( validate model on test set )
with torch.no_grad():   # turn off back propagation 
    independent_eval = model.forward(dependent_test) # test are features from our test set, and the eval will be predictions
    loss = criterion(independent_eval, independent_test) # find the loss or error from ind eval vs ind test
    #print(loss)

correct = 0
with torch.no_grad():
    for i, data in enumerate(dependent_test):
        independent_eval = model.forward(data)

        print(f'{i+1}.) {str(independent_eval)} \t {independent_test[i]} \t {independent_eval.argmax().item()}')

        # correct or not 
        if independent_eval.argmax().item() == independent_test[i]:
            correct += 1

print(f'We have {correct} correct')

# pass metrics to wandb or mlflow so we can evaluate what da heck is going on <3 