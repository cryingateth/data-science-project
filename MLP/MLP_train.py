import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from Architecture import MLP
from Dataset import BRCA_Dataset

# Constants
CSV_FILE_PATH = 'Dataset/balanced.csv'
TARGET_COLUMN = 'BRCA_subtype'
TEST_SIZE = 0.3
VALIDATION_TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50
MODEL_PATH = 'best_model.pt'

# Function to calculate loss
def calculate_loss(loader, model):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    total_samples = 0
    correct_samples = 0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
    return correct_samples / total_samples

# Load and process data
data = pd.read_csv(CSV_FILE_PATH)
features = data.drop(columns=[TARGET_COLUMN])
numerical_features = features.select_dtypes(include=[np.number])

le = LabelEncoder()
target = le.fit_transform(data[TARGET_COLUMN])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features)

X_train, X_val_test, y_train, y_val_test = train_test_split(scaled_features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=VALIDATION_TEST_SIZE, random_state=RANDOM_STATE)

# Prepare data loaders
train_dataset = BRCA_Dataset(X_train, y_train)
val_dataset = BRCA_Dataset(X_val, y_val)
test_dataset = BRCA_Dataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model instantiation
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

model = MLP(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = np.inf
train_losses = []
val_losses = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_loss = calculate_loss(val_loader, model)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)

    train_loss = calculate_loss(train_loader, model)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

# Load best model weights
model.load_state_dict(best_model_wts)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load best model weights
model.load_state_dict(best_model_wts)

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted.numpy().tolist())

print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_true, y_pred)}')
print(f'Classification Report: \n{classification_report(y_true, y_pred)}')

