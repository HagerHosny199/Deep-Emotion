import torch
import torch.nn as nn
import torch.nn.functional as F

class Deep_Emotion(nn.Module):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,1,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(81,50)
        self.fc2 = nn.Linear(50,7)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 2*3)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        #print('xs.size()',xs.size())
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        #print('theta',theta.size())

        theta = theta.view(-1, 2, 3)
        #print(' x.size()', x.size())
        grid = F.affine_grid(theta,[x.size()[0], 1, 9, 9])
        #print('grid', grid.size())
        #x = F.grid_sample(x, grid)
        return grid

    def forward(self,input):
        out1 = self.stn(input)
        #print('out1.size():',out1.size())

        out = F.relu(self.conv1(input))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        out = F.relu(self.pool4(out))
        #print('out.size():',out.size())
        
        out = F.dropout(out)
        third_tensor = F.grid_sample(out, out1)
        #print('third_tensor.size():',third_tensor.size())
        third_tensor = third_tensor.view(-1, 81)
        #out1 = out1.view(-1, 81)
        #out1 = out1.view(-1, 810)
        #print('out.size():',out.size())
        #print('out1.size():',out1.size())
        
        #print('third_tensor.size():',third_tensor.size())
        out = F.relu(self.fc1(third_tensor))
        out = self.fc2(out)

        return out
