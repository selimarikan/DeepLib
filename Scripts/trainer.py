import argparse
import torch
import torchvision
import numpy as np

def ProcessArguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--dataset", required = True, type = str, help = "Path to the dataset images")
    argParser.add_argument("--model", required = True, type = str, help = "Model to use for training")
    argParser.add_argument("--optimizer", required = True, type = str, help = "Optimizer to be used")
    argParser.add_argument("--epochs", required = True, type = int, help = "How many epochs for training")
    argParser.add_argument("--imagesize", required = True, type = int, help = "Size of the image for training")

class LearningConfig(object):
    def __init__(self, datasetPath, modelName, optimizerName, optimizerState, epoch, imageSize, batchSize, lr, l2, lrDecay):
        self.datasetPath = datasetPath
        self.modelName = modelName
        self.optimizerName = optimizerName
        self.optimizerState = optimizerState
        self.epoch = epoch
        self.imageSize = imageSize
        self.batchSize = batchSize
        self.lr = lr
        self.l2 = l2
        self.lrDecay = lrDecay
    
    def GetDict(self):
        return {
        'DatasetPath' : self.datasetPath,
        'ModelName' : self.modelName,
        'OptimizerName' : self.optimizerName,
        'OptimizerState' : self.optimizerState,
        'Epoch' : self.epoch,
        'ImageSize' : self.imageSize,
        'BatchSize' : self.batchSize,
        'LearningRate' : self.lr,
        'L2' : self.l2,
        'LRDecay' : self.lrDecay,
        }

def CreateConfig():
    configDict = {
        


    }


if __name__ == "__main__":
    ProcessArguments()