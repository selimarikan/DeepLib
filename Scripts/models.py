import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfNet(nn.Module):
    def __init__(self, classCount, imageSize):
        super(SurfNet, self).__init__()
        self.classCount = classCount
        self.imageSize = imageSize

        self.convIn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNIn = nn.BatchNorm2d(num_features=16)
        self.reluIn = nn.PReLU()
        
        self.convIA = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIA = nn.BatchNorm2d(num_features=16)
        self.reluIA = nn.PReLU()
                
        self.convI2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI2 = nn.BatchNorm2d(num_features=32)
        self.reluI2 = nn.PReLU()
        
        self.convIB = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIB = nn.BatchNorm2d(num_features=32)
        self.reluIB = nn.PReLU()
        
        self.convI3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), stride=(2,2), padding=(2,2), dilation=(1,1))
        self.batNI3 = nn.BatchNorm2d(num_features=64)
        self.reluI3 = nn.PReLU()
        
        self.convIC = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNIC = nn.BatchNorm2d(num_features=64)
        self.reluIC = nn.PReLU()
        
        # Extended nonlinearity
        self.convI4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI4 = nn.BatchNorm2d(num_features=64)
        self.reluI4 = nn.PReLU()
        
        self.convI5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI5 = nn.BatchNorm2d(num_features=64)
        self.reluI5 = nn.PReLU()
        
        self.convI6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=(1,1))
        self.batNI6 = nn.BatchNorm2d(num_features=64)
        self.reluI6 = nn.PReLU()
            
        self.outMul = int(self.imageSize / 8) 
            
        self.fc = nn.Sequential(
            nn.Linear(64 * self.outMul * self.outMul, self.classCount),
            )
        self.logsmax = nn.LogSoftmax()

    def forward(self, x):
        # Convolutional layer 1
        convInResult = self.convIn(x)
        #convInResult.register_hook(cnnUtils.save_grad('convInGrad'))
        #  Add the residual before activation function
        x = self.reluIn(self.batNIn(convInResult)) #  + plus01
        resIn = x
        
        x = self.reluIA(self.batNIA(self.convIA(x)) + resIn)
        
        # Convolutional layer 2
        convI2Result = self.convI2(x)
        #convI2Result.register_hook(save_grad('convI2Grad'))
        #  Add the residual before activation function
        x = self.reluI2(self.batNI2(convI2Result)) # + plus02
        resI2 = x
        
        x = self.reluIB(self.batNIB(self.convIB(x)) + resI2)
        
        # Convolutional layer 3
        convI3Result = self.convI3(x)
        #convI3Result.register_hook(save_grad('convI3Grad'))
        #  Add the residual before activation function
        x = self.reluI3(self.batNI3(convI3Result)) #  + plus03
        resI3 = x
        
        x = self.reluIC(self.batNIC(self.convIC(x)) + resI3)
        
        x = self.reluI4(self.batNI4(self.convI4(x)))
        resI4 = x
        x = self.reluI5(self.batNI5(self.convI5(x)) + resI4)
        resI5 = x
        x = self.reluI6(self.batNI6(self.convI6(x)) + resI5)
        
        # Reshape the result for fully-connected layers
        x = x.view(-1, 64 * self.outMul * self.outMul)
        
        # Apply the result to fully-connected layers
        x = self.fc(x)
        
        # Finally apply the LogSoftMax for output
        x = self.logsmax(x)
        return x