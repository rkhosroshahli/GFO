import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

from model import NeuralNetwork
from data_loader import *
from gfo import GradientFreeOptimization

import matplotlib.pyplot as plt
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# train_loader = load_mnist_train(samples_per_class=100, seed=42, batch_size=128)
train_loader = load_mnist_train_full(batch_size=128)
test_loader = load_mnist_test(batch_size=128)

gfo = GradientFreeOptimization(NeuralNetwork, train_loader, DEVICE)
gfo_test = GradientFreeOptimization(NeuralNetwork, test_loader, DEVICE)
model = gfo.model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Train the network
num_epochs = 10
for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            true_labels = []
            predicted_labels = []

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                
                true_labels.extend(target.tolist())
                predicted_labels.extend(predicted.tolist())

            train_loss = running_loss / len(train_loader)
            train_f1score = f1_score(true_labels, predicted_labels, average='macro')

            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Training F1-score: {train_f1score*100:.2f}%')

params = gfo.get_parameters(model)
dimensions = len(params)
train_f1score = gfo.fitness_func(params) * -100
test_f1score = gfo_test.fitness_func(params) * -100

f1_train_after_block, f1_test_after_block = [], []

df = pd.DataFrame({"Data": ["Training", "Test"], f"No blocking D={dimensions}": [train_f1score, test_f1score]})

new_blocked_dimensions = []
selected_blocked_dimensions= np.arange(start=100, stop=10000, step=100)
for i in range(len(selected_blocked_dimensions)):
    blocked_dimensions = selected_blocked_dimensions[i]
    print("Interval:", blocked_dimensions)
    
    # f1 = gfo.fitness_func(params) * -100
    # f1score_before_block.append(f1)
    # print("F1-score after Adam", f1)

    # Define the number of bins
    num_bins = blocked_dimensions

        # Calculate the bin edges
    bin_edges = np.linspace(params.min(),params.max(), num_bins)

        # Split the data into bins
    binned_data = np.digitize(params, bin_edges)

    new_blocked_dims=0
    blocks=[]
    for i in range(blocked_dimensions):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) != 0:
                new_blocked_dims+=1
                blocks.append(b_i)
    print("Optimal blocked dimensions",new_blocked_dims)
    new_blocked_dimensions.append(new_blocked_dims)
    # blocking
    params_blocked = np.zeros(new_blocked_dims)
    for i in range(new_blocked_dims):
            block_params = params[blocks[i]].copy()
            if len(block_params) != 0:
                params_blocked[i] = np.mean(block_params)

    # unblocking
    params_unblocked=np.ones(dimensions)
    for i in range(new_blocked_dims):
            params_unblocked[blocks[i]] *= params_blocked[i]

    
    train_f1score = gfo.fitness_func(params_unblocked) * -100
    f1_train_after_block.append(train_f1score)
    print("Training F1-score after blocking and unblocking", train_f1score)
    test_f1score = gfo_test.fitness_func(params_unblocked) * -100
    f1_test_after_block.append(test_f1score)
    print("Test F1-score after blocking and unblocking", test_f1score)

    df[f"Blocked w/ Interval={blocked_dimensions}, D`={new_blocked_dims}"] = [train_f1score, test_f1score]

    print("-"*50)

df.to_csv("ann_mnist_optimal_blocking_f1score.csv", index=False)

print("Optimal blocked dimensions:", new_blocked_dimensions)

plt.figure(figsize=(12, 5))
plt.plot(selected_blocked_dimensions, f1_train_after_block, label="Train")
plt.plot(selected_blocked_dimensions, f1_test_after_block, label="Test")
plt.xticks(selected_blocked_dimensions)
plt.legend()
plt.grid()
plt.xlabel("F1-score")
plt.ylabel("Intervals")
plt.title("ANN, MNIST")
plt.savefig("ann_mnist_optimal_blocking_plot.png")
plt.show()