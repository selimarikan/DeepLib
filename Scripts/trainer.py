import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

configPath = './config.json'
appConfig = None

class JSONClass(object):
    def __init__(self, jsonStr):
        self.__dict__ = json.loads(jsonStr)

def LoadConfig(configPath):
    f = open(configPath, 'r')
    configStr = f.readlines()
    return JSONClass(configStr)
    

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


def CreateTrafos(imageSize, cropSize, setMean, setStd, onlineAug = False):
    trafos = {
        'train': transforms.Compose([
            transforms.Random
        ])
    }

def CreateDataset(dataDir, trafo):
    datasets = {x: datasets.ImageFolder(os.path.join(dataDir, x), trafo[x]) for x in ['train', 'test']}


def Train():
    pass








def CreateConfig():
    configDict = {
        


    }


if __name__ == "__main__":
    ProcessArguments()