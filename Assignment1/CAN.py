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


class CAN(torch.nn.Module):

    def __init__(self):
        super(CAN, self).__init__()

        # Feature map size = 28+2*padding-(2*dilation+1)+1 = 28
        # so the size of feature maps retain the size of input image, ie: 28
        self.conv1 = torch.nn.Conv2d(1, 32, 3, dilation=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, dilation=2, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, dilation=4, padding=4)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, dilation=8, padding=8)
        self.conv5 = torch.nn.Conv2d(32, 10, 3, dilation=1, padding=1)
        self.pool = torch.nn.AvgPool2d(28)

    def forward(self, X):
        X = torch.nn.functional.leaky_relu(self.conv1(X))
        X = torch.nn.functional.leaky_relu(self.conv2(X))
        X = torch.nn.functional.leaky_relu(self.conv3(X))
        X = torch.nn.functional.leaky_relu(self.conv4(X))
        X = torch.nn.functional.leaky_relu(self.conv5(X))
        X = self.pool(X)
        X = torch.flatten(X, 1)
        return X


can = CAN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(can.parameters(), lr=0.01)

# the trainning loop
for epoch in range(num_epoch):  # 20 epochs
    can.train()
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
        prediction = can(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        train_loss_in_this_epoch += loss.item()*this_batch_size

    print("epoch", epoch, end=": ")
    print("the loss is", train_loss_in_this_epoch/60000)

test_prediction = can(full_test_data)
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
