import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from seqInit import toTs
from seqInit import input_size
from seqInit import train, val
from model import gruModel, lstmModel, rnnModel
import numpy as np

# set model & log_dir
model = lstmModel(1, 5, 1, 2).cuda(0)
writer = SummaryWriter('./logs/lstm')

# set loss function and optimization function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Input processing
train = train.reshape(-1, 1, 1)
x = torch.from_numpy(train[:-1]).cuda(0)
y = torch.from_numpy(train[1:])[12:].cuda(0)

# train model
frq, sec = 4000, 100
loss_set = []

for e in range(1, frq + 1):
    inputs = Variable(x).cuda(0)
    target = Variable(y).cuda(0)

    # forward
    output = model(inputs)
    loss = criterion(output, target)

    # update paramters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print training information
    print_loss = loss.item()
    loss_set.append((e, print_loss))
    if e % sec == 0:
        writer.add_scalar('Train/Loss', print_loss, e)
        print('Epoch[{}/{}], Loss: {:.5f}'.format(e, frq, print_loss))


model = model.eval()

# 预测结果并比较

val = val.reshape(-1, 1, 1)
px = val[:-1]
px = torch.from_numpy(px)
ry = torch.from_numpy(val[1:][12:])
varX = Variable(px).cuda(0)
py = model(varX).data.cpu()
Loss_val = criterion(py, ry)
print('Loss: ', Loss_val)

writer.close()
