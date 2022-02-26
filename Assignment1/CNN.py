import torch
import numpy as np
from torchvision.transforms.functional import to_tensor
from FileLoader import *

batch_size = 64
num_epoch = 20


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

full_train_data = full_train_data.reshape(60000, 1, 28, 28)
full_test_data = full_test_data.reshape(10000, 1, 28, 28)


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # feature map will have size 24*24
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.avgPool1 = torch.nn.AvgPool2d(2, 2)  # down sampled to size 12*12
        self.conv2 = torch.nn.Conv2d(6, 16, 5)  # featur map will have size 8*8
        self.avgPool2 = torch.nn.AvgPool2d(2, 2)  # down sampled to size 4*4

        self.fc1 = torch.nn.Linear(16 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, X):
        X = self.conv1(X)
        X = self.avgPool1(X)
        X = self.conv2(X)
        X = self.avgPool2(X)
        X = torch.flatten(X, 1)
        X = self.fc1(X)
        X = torch.tanh(X)
        X = self.fc2(X)
        X = torch.tanh(X)
        X = self.fc3(X)
        return X


cnn = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.01)


# the trainning loop
for epoch in range(num_epoch):  # 20 epochs
    cnn.train()
    permutation = torch.randperm(60000)
    train_loss_in_this_epoch = 0.0

    # batch size of 64, takes 938 iterations to go through whole dataset
    for batch in range(60000//batch_size + 1):
        if batch != 60000//batch_size:
            this_batch_size = batch_size
            input = np.zeros(batch_size*28*28).reshape(batch_size, 1, 28, 28)
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
                             28).reshape(this_batch_size, 1, 28, 28)
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
        prediction = cnn(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        train_loss_in_this_epoch += loss.item()*this_batch_size

    print("epoch", epoch, end=": ")
    print("the loss is", train_loss_in_this_epoch/60000)

test_prediction = cnn(full_test_data)
test_loss = criterion(test_prediction, full_test_target)
print("the loss on test data is", test_loss.item())

predicted_numbers = []
for i in range(10000):
    predicted_number = torch.argmax(test_prediction[i]).item()
    predicted_numbers.append(predicted_number)
predicted_numbers = np.array(predicted_numbers)

num_error = 0
for truth, prediction in zip(test_labels, predicted_numbers):
    if truth != prediction:
        num_error += 1
print("The accuracy on the testing data is", 1-num_error/10000)

"""
the loss on test data is 0.09200327463574685
The accuracy on the testing data is 0.9716
"""