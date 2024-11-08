from __future__ import print_function
import torch.nn as nn
import time
from datetime import date, datetime
from Infor import infor
from torch.optim import Adam
from imageloader import *
import imagemodel

inf = infor()
inf.modelname = 'Standard LeNet Neuron Network'
inf.datetime = str(date.today().strftime("%B %d, %Y")) + ',  ' + str(datetime.now())
inf.description = 'Test of ADAM optimizer on Mitbih data set'
inf.TorchNumberThreads = 4
inf.LossFunction = 'CrossEntropyLoss'
inf.INIT_LR = 1e-4
inf.BATCHSIZE = 32
inf.EPOCH = 20
inf.trainset = './DATA/NEU-DET/TRAIN'
inf.testset = './DATA/NEU-DET/VAL'
ResFolder = 'DATA'
inf.nbclasses = 6

train_data, test_data = getimages(inf.trainset, inf.testset)

startTime = time.time()
model = imagemodel.LeNet()
opt = Adam(model.parameters(), lr=inf.INIT_LR)
lossFn = nn.CrossEntropyLoss()
for e in range(inf.EPOCH):
    print(f'Đang train tới epoch thứ {e}', end='. ')
    model.train()
    totalTrainLoss = 0
    trainCorrect = 0
    # loop over the training set
    for (x, y) in train_data:
        pred = model(x)
        loss = lossFn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        totalTrainLoss += loss
    print(f'Total train loss = {totalTrainLoss}')
# finish measuring how long training took
endTime = time.time()
inf.TrainTime = endTime - startTime
print(f'Training time: {inf.TrainTime}')
correct = 0
total = 0
for images, labels in test_data:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100}%')




