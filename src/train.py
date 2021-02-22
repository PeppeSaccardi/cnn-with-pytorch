# train.py

import torch
import config
import argparse
import random
import model
import os 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
import torch.optim as optim


# Function that get the input from the user
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser
    

def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def run(lr, BATCH_SIZE, EPOCHS):
    
    df_train = pd.read_csv(config.TRAIN_DATA, index_col=0)
    df_valid = pd.read_csv(config.VALID_DATA, index_col=0)

    y_train = df_train.target.values
    df_train.pop('target')
    X_train = df_train
    training_set = model.ShapeDataset(
        features = X_train,
        targets = y_train
    )
    
    y_valid = df_valid.target.values
    df_valid.pop('target')
    X_valid = df_valid
    valid_set = model.ShapeDataset(
        features = X_valid,
        targets = y_valid
    )
    
    training_loader = dataloader.DataLoader(
        dataset=training_set, batch_size=BATCH_SIZE, shuffle=True
    )

    validation_loader = dataloader.DataLoader(
        dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False
    ) 


    data_loaders = {
        "train" : training_loader,
        "val" : validation_loader
        }

    data_lengths = {
        "train": len(y_train), 
        "val": len(y_valid)
        }

    train_running_loss = []
    train_acc = []


    net = model.Model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        params=net.parameters(), lr=lr
        )

    losses = {
        "train":[], "val": []
        }

    accuracies = {
        "train":[], "val": []
    }

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        net = net.double()
        
        
        for phase in ['train', 'val']:

            
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
                
            correct = 0
            iter_loss = 0
            for i, (inputs, labels) in enumerate(data_loaders[phase], 0):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                iter_loss += loss.item()
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

            losses[phase].append(iter_loss/(i+1))
            accuracies[phase].append(100 * correct / data_lengths[phase])

            if phase == "train":
                train_accuracy = accuracies["train"]
                print(
                    "Epoch [{}/{}], Training loss: {:.3f}, Training accuracy: {:.3f}".format(
                        epoch+1, EPOCHS, losses[phase][-1], train_accuracy[-1]
                        )
                    )
            else:
                print(
                    "Epoch [{}/{}], Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(
                        epoch+1, EPOCHS, losses[phase][-1], accuracies[phase][-1]
                        )
                    )

    train_acc = accuracies["train"]
    val_acc = accuracies["val"]
    loss_train = losses["train"]
    loss_valid = losses["val"]
    plt.figure(1)
    plt.subplot(121)
    plt.plot(loss_train, label = "Train loss")
    plt.plot(loss_valid, label = "Valid loss")
    plt.title("Losses")
    plt.subplot(122)
    plt.plot(train_acc, label = "Train accuracy")
    plt.plot(val_acc, label = "Valid accuracy")
    plt.title("Accuracies")
    
    if "result" not in os.listdir("../documents/"):
        os.makedirs("../documents/result")
        plt.savefig("../documents/result/image.png")
        accuracies = pd.DataFrame(accuracies)
        losses = pd.DataFrame(losses)
        accuracies.to_csv(config.OUTPUT_FOLD + "accuracies.csv")
        losses.to_csv(config.OUTPUT_FOLD + "losses.csv")
        with open("config.py", "r") as f:
            data = f.readlines()
            with open("config.py","w") as file:
                for line in data:
                    file.write(line)
                file.write("\n")
                file.write("""ACCURACIES = "../output/accuracies.csv" """+"\n")
                file.write("""LOSSES = "../output/losses.csv" """)
                file.close()
            f.close()
        



if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    
    set_all_seeds(seed=args.seed)
    run(lr=args.lr, BATCH_SIZE=args.bs, EPOCHS=args.epochs)
    