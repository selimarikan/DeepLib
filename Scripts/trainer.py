import argparse
import torch
import torchvision
import numpy as np

def HandleArguments():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--dataset", required = True, type = str, help = "Path to the dataset images")
    argParser.add_argument("--model", required = True, type = str, help = "Model to use for training")
    argParser.add_argument("--optimizer", required = True, type = str, help = "Optimizer to be used")
    argParser.add_argument("--epochs", required = True, type = int, help = "How many epochs for training")
    argParser.add_argument("--imagesize", required = True, type = int, help = "Size of the image for training")


if __name__ == "__main__":
    HandleArguments()