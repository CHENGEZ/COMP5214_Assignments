import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from FileLoader import *

batch_size = 64
num_epoch = 20
# adjust to plot a curve of accuracy versus the number of neurons: 4, 8, 16, 32, 64, 128, and 256
num_of_neurons_in_hidden = 256


train_images, train_labels = load_train_data()
full_train_data = to_tensor(train_images)[0, :, :]
full_train_target = []
for i in range(60000):
    full_train_target.append(np.zeros(10))
for i in range(60000):
    k = train_labels[i]
    full_train_target[i][k] = 1
full_train_target = np.array(full_train_target)
full_train_target = torch.from_numpy(full_train_target)


test_images, test_labels = load_test_data()
full_test_data = to_tensor(test_images)[0, :, :]
full_test_target = []
for i in range(10000):
    full_test_target.append(np.zeros(10))
for i in range(10000):
    k = test_labels[i]
    full_test_target[i][k] = 1
full_test_target = np.array(full_test_target)
full_test_target = torch.from_numpy(full_test_target)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fcl1 = torch.nn.Linear(28*28, num_of_neurons_in_hidden)
        self.fcl2 = torch.nn.Linear(
            num_of_neurons_in_hidden, num_of_neurons_in_hidden)
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
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

# training loop
for epoch in range(num_epoch):  # 20 epochs
    mlp.train()
    permutation = torch.randperm(60000)
    train_loss_in_this_epoch = 0.0

    # batch size of 64, takes 938 iterations to go through whole dataset
    for batch in range(60000//batch_size + 1):
        # print("batch", batch,end=": ")
        if batch != 60000//batch_size:
            this_batch_size = batch_size
            input = np.zeros(batch_size*28*28).reshape(batch_size, 28*28)
            input = torch.from_numpy(input)
            for i in range(batch_size):
                input[i] = full_train_data[permutation[batch*batch_size+i]]
            input = input.to(torch.float32)

            target = np.zeros(batch_size*10).reshape(batch_size, 10)
            target = torch.from_numpy(target)
            for i in range(batch_size):
                target[i] = full_train_target[permutation[batch*batch_size+i]]
            target = target.to(torch.float32)

        else:
            this_batch_size = 60000-batch*batch_size
            input = np.zeros(this_batch_size*28 *
                             28).reshape(this_batch_size, 28*28)
            for i in range(this_batch_size):
                input[i] = full_train_data[permutation[batch*batch_size+i]]
            input = torch.from_numpy(input)
            input = input.to(torch.float32)

            target = np.zeros(this_batch_size*10).reshape(this_batch_size, 10)
            for i in range(this_batch_size):
                target[i] = full_train_target[permutation[batch*batch_size+i]]
            target = torch.from_numpy(target)
            target = target.to(torch.float32)

        optimizer.zero_grad()
        prediction = mlp(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        train_loss_in_this_epoch += loss.item()*this_batch_size

    print("epoch", epoch, end=": ")
    print("the loss is", train_loss_in_this_epoch/60000)

test_prediction = mlp(full_test_data)
test_loss = criterion(test_prediction, full_test_target)
print("the loss on test data is", test_loss.item())


"""
test accuract depending on differnet num_of_neurons_in_hidden
4: the loss on test data is 0.5520997686341712
8: the loss on test data is 0.2870218555975418
16: the loss on test data is 0.19988575598033148
32: the loss on test data is 0.165200721723496
64: the loss on test data is 0.15033791905283514
128: the loss on test data is 0.14072852918090104
256: the loss on test data is 0.13293400765137472

without trainning: the loss on test data is 2.303080479717255
"""
