import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from Architecture import MLP
from Dataset import BRCA_Dataset

# Constants
CSV_FILE_PATH = '../Dataset/balanced.csv'
TARGET_COLUMN = 'BRCA_subtype'
TEST_SIZE = 0.3
VALIDATION_TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_EPOCHS = 50
MODEL_PATH = 'best_model.pt'

# Possible hyperparameters
LEARNING_RATES = [0.1, 0.01, 0.001]
BATCH_SIZES = [32, 64, 128]

# Function to calculate loss
def calculate_loss(loader, model):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

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

# Model instantiation
input_size = X_train.shape[1]
output_size = len(np.unique(y_train))

best_model_wts = None
best_loss = np.inf
best_lr = None
best_batch_size = None

for LEARNING_RATE in LEARNING_RATES:
    for BATCH_SIZE in BATCH_SIZES:
        # Model instantiation
        model = MLP(input_size, output_size)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training loop
        train_losses = []
        val_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = calculate_loss(train_loader, model)
            val_loss = calculate_loss(val_loader, model)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_lr = LEARNING_RATE
                best_batch_size = BATCH_SIZE
                torch.save(model.state_dict(), MODEL_PATH)

        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Losses with LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'loss_plot_lr{LEARNING_RATE}_bs{BATCH_SIZE}.png')

        print(f'Finished Training with LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Best Validation Loss: {best_loss}')

print(f'Best Hyperparameters - Learning Rate: {best_lr}, Batch Size: {best_batch_size}')

# Load best model weights
model.load_state_dict(best_model_wts)

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predicted.cpu().numpy().tolist())

print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
print(f'Confusion Matrix: \n{confusion_matrix(y_true, y_pred)}')
print(f'Classification Report: \n{classification_report(y_true, y_pred)}')
