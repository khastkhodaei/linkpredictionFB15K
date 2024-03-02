import os
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset  # import TrainDataset class from dataset.py
from models import TransE

# Define the data directory
DATA_DIR = "./data/FB15k/"
SAVED_MODELS_DIR = "./saved_models/"

# Ensure that the directory for saved models exists
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def load_data(filename):
    """
    Read data from a file and convert it to a pandas DataFrame.
    """
    file_path = DATA_DIR + filename
    data = pd.read_csv(file_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    return data


def preprocess_data(data):
    """
    Preprocess the data for model usage.
    """
    label_encoder = LabelEncoder()
    data['head'] = label_encoder.fit_transform(data['head'])
    data['tail'] = label_encoder.fit_transform(data['tail'])
    data['relation'] = label_encoder.fit_transform(data['relation'])
    return data


def data_check(data):
    """
    Test data values to ensure preprocessing correctness.
    """
    print("Number of rows in data:", len(data))
    print("\nData types and values:")
    print(data.dtypes)
    print(data.head())


def train_model(model, criterion, optimizer, train_loader, num_epochs, device):
    """
    Train the model using the training data.
    """
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            head = batch['head'].to(device)
            relation = batch['relation'].to(device)
            tail = batch['tail'].to(device)

            # Forward pass for positive samples
            positive_distance = model(head, relation, tail)

            # Calculate the margin loss
            loss = criterion(positive_distance, torch.zeros_like(positive_distance),
                             torch.ones_like(positive_distance).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), "./saved_models/trained_model_Fb15K.pth")


if __name__ == "__main__":
    train_data = load_data("train.txt")
    valid_data = load_data("valid.txt")
    test_data = load_data("test.txt")

    print("Before Preprocessing:")
    print("\nTrain Data:")
    data_check(train_data)
    print("\nValid Data:")
    data_check(valid_data)
    print("\nTest Data:")
    data_check(test_data)

    train_data = preprocess_data(train_data)
    valid_data = preprocess_data(valid_data)
    test_data = preprocess_data(test_data)

    print("\n\nAfter Preprocessing:")
    print("\nTrain Data:")
    data_check(train_data)
    print("\nValid Data:")
    data_check(valid_data)
    print("\nTest Data:")
    data_check(test_data)

    num_entities = 15003  # Based on the recommendation
    num_relations = 1345  # Based on the recommendation
    embedding_dim = 100

    model = TransE(num_entities, num_relations, embedding_dim)

    criterion = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Creating train dataset and dataloader
    train_dataset = TrainDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    model.to(device)

    train_model(model, criterion, optimizer, train_loader, num_epochs=10, device=device)
