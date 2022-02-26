import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from FileLoader import *

batch_size = 64
num_epoch = 20
num_of_neurons_in_hidden = 128 # adjust to plot a curve of accuracy versus the number of neurons: 4, 8, 16, 32, 64, 128, and 256


train_images, train_labels = load_train_data()
full_train_data = to_tensor(train_images)[0,:,:]
full_train_target = []
for i in range(60000):
    full_train_target.append(np.zeros(10))
for i in range(60000):
    k = train_labels[i]
    full_train_target[i][k] = 1
full_train_target = np.array(full_train_target)
full_train_target = torch.from_numpy(full_train_target)


test_images, test_labels = load_test_data()
full_test_data = to_tensor(test_images)[0,:,:]
full_test_target = []
for i in range(10000):
    full_test_target.append(np.zeros(10))
for i in range(10000):
    k = train_labels[i]
    full_test_target[i][k] = 1
full_test_target = np.array(full_test_target)
full_test_target = torch.from_numpy(full_test_target)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fcl1 = torch.nn.Linear(28*28, num_of_neurons_in_hidden)
        self.fcl2 = torch.nn.Linear(num_of_neurons_in_hidden, num_of_neurons_in_hidden)
        self.fcl3 = torch.nn.Linear(num_of_neurons_in_hidden, 10)

    def forward(self, X):
        X = self.fcl1(X)
        X = torch.nn.functional.relu(X)
        X = self.fcl2(X)
        X = torch.nn.functional.relu(X)
        X = self.fcl3(X)
        return X

mlp = MLPClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(),lr=0.005)

# training loop
for epoch in range(num_epoch): # 20 epochs
    mlp.train()
    permutation = torch.randperm(60000)
    print("epoch", epoch, end=":")

    for batch in range(60000//batch_size + 1): # batch size of 64, takes 938 iterations to go through whole dataset
        if batch != 60000//batch_size:
            input = full_train_data[permutation[batch*batch_size]:permutation[(batch+1)*batch_size]]
            target = full_train_target[permutation[batch*batch_size]:permutation[(batch+1)*batch_size]]
            this_batch_size = batch_size
        else:
            input= full_train_data[permutation[batch*batch_size]:]
            target = full_train_target[permutation[batch*batch_size]:]
            this_batch_size = 60000-batch*batch_size
        optimizer.zero_grad()
        prediction = mlp(input)
        loss = criterion(prediction,target)
        loss.backward()
        optimizer.step()
    print("the loss is",loss/this_batch_size)

test_prediction = mlp(full_test_data)
test_loss = criterion(test_prediction, full_test_target)
print("the loss on test data is", test_loss/10000)