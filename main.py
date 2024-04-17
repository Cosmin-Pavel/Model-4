
from torch.utils.data import DataLoader,Dataset
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import time
from constants import FILE_PATH, SKIP_ROWS, NUM_ROWS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import math

def import_chess_data(file_path, skip_rows=0, num_rows=None):
    try:
        data = pd.read_csv(file_path, skiprows=range(1, skip_rows + 1), nrows=num_rows)
        return data
    except FileNotFoundError:
        print("File not found. Please make sure the file path is correct.")
        return None


def preprocess_data1(data):
    fen_data = data['FEN']
    evaluation_data = data['Evaluation']
    return fen_data, evaluation_data


def preprocess_data2(data):
    fen_data = data['FEN']
    evaluation_data = data['Evaluation']

    # Process evaluation data
    processed_evaluation = []
    for value in evaluation_data:
        try:
            # Attempt to convert value to integer
            int_value = int(value)
            processed_evaluation.append(int_value)
        except ValueError:
            # If conversion fails, check if second character is '+'
            if len(value) >= 2 and value[1] == '+':
                processed_evaluation.append(10000)
            else:
                processed_evaluation.append(-10000)

    # Convert processed_evaluation to a Pandas Series
    processed_evaluation_series = pd.Series(processed_evaluation)

    return fen_data, processed_evaluation_series


def normalize_evaluation(evaluation_data):
    min_val = evaluation_data.min()
    max_val = evaluation_data.max()
    normalized_data = (evaluation_data - min_val) / (max_val - min_val)
    return normalized_data


def fen_to_bit_vector(fen):
    parts = re.split(" ", fen)
    piece_placement = re.split("/", parts[0])
    active_color = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_clock = int(parts[5])

    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)

    # piece to layer structure taken from reference [1]
    piece_to_layer = {
        'R': 1,
        'N': 2,
        'B': 3,
        'Q': 4,
        'K': 5,
        'P': 6,
        'p': 7,
        'k': 8,
        'q': 9,
        'b': 10,
        'n': 11,
        'r': 12
    }

    castling = {
        'K': (7, 7),
        'Q': (7, 0),
        'k': (0, 7),
        'q': (0, 0),
    }

    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)

    if en_passant != '-':
        bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1

    if castling_rights != '-':
        for char in castling_rights:
            bit_vector[0, castling[char][0], castling[char][1]] = 1

    if active_color == 'w':
        bit_vector[0, 7, 4] = 1
    else:
        bit_vector[0, 0, 4] = 1

    if halfmove_clock > 0:
        c = 7
        while halfmove_clock > 0:
            bit_vector[0, 3, c] = halfmove_clock % 2
            halfmove_clock = halfmove_clock // 2
            c -= 1
            if c < 0:
                break

    if fullmove_clock > 0:
        c = 7
        while fullmove_clock > 0:
            bit_vector[0, 4, c] = fullmove_clock % 2
            fullmove_clock = fullmove_clock // 2
            c -= 1
            if c < 0:
                break

    return bit_vector


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(832, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def train_model(model, criterion, optimizer, train_loader, test_loader, device,batch_size,train_size, test_size,num_epochs=10):
    train_losses_mse = []
    train_losses_mae = []
    train_maes = []

    test_losses_mse = []  # Store test losses per epoch (MSE)
    test_losses_mae = []  # Store test losses per epoch (MAE)
    test_maes = []  # Store test MAEs per epoch


    for epoch in range(num_epochs):
        model.train()
        running_loss_mse = 0.0
        running_loss_mae = 0.0
        running_mae = 0.0
        total_samples = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_mse = criterion(outputs, labels.float().unsqueeze(1))
            loss_mae = torch.mean(torch.abs(outputs.squeeze(1) - labels))
            absolute_errors = torch.abs(outputs.squeeze(1) - labels)

            running_loss_mse += loss_mse.item() * inputs.size(0)
            running_loss_mae += loss_mae.item() * inputs.size(0)
            running_mae += torch.sum(absolute_errors).item()
            total_samples += labels.size(0)
            loss_mse.backward()
            optimizer.step()


        epoch_loss_mse = running_loss_mse / total_samples
        epoch_loss_mae = running_loss_mae / total_samples
        epoch_mae = running_mae / total_samples
        train_losses_mse.append(epoch_loss_mse)
        train_losses_mae.append(epoch_loss_mae)
        train_maes.append(epoch_mae)


        # Evaluate model on test data
        test_loss_mse, test_loss_mae, test_mae = evaluate_model(model, criterion, test_loader, device)
        test_losses_mse.append(test_loss_mse)
        test_losses_mae.append(test_loss_mae)
        test_maes.append(test_mae)


        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss (MSE): {epoch_loss_mse:.4f}, Train Loss (MAE): {epoch_loss_mae:.4f} \t Test Loss (MSE): {test_loss_mse:.4f}, Test Loss (MAE): {test_loss_mae:.4f}")

    plot_learning_curves(train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae, optimizer, num_epochs,batch_size,train_size, test_size)

    return train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae, train_maes


def plot_learning_curves(train_losses_mse, test_losses_mse, train_losses_mae, test_losses_mae, optimizer, num_epochs, batch_size,train_size, test_size):
    plt.figure(figsize=(12, 6))

    # Plotting MSE learning curves
    plt.plot(range(1, num_epochs + 1), train_losses_mse, label='Training Loss (MSE)')
    plt.plot(range(1, num_epochs + 1), test_losses_mse, label='Test Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MSE Learning Curves')
    plt.legend()

    # Add model information as text on the right side
    text = f"Optimizer: {optimizer.__class__.__name__}\nEpochs: {num_epochs}\nBatch Size: {batch_size}\nLearning Rate: {optimizer.param_groups[0]['lr']}\nTraining Data Size: {train_size}\nTest Data Size: {test_size}"
    plt.text(1.02, 0.5, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Adjust layout to fit the text and plot within the image
    plt.subplots_adjust(right=0.7)  # Adjust the space on the right for the text
    plt.margins(0.2)  # Add some margin around the plot

    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_mse = f'plots/mse_learning_curves_{timestamp}.png'
    plt.savefig(filename_mse)
    plt.show()

    plt.figure(figsize=(12, 6))

    # Plotting MAE learning curves
    plt.plot(range(1, num_epochs + 1), train_losses_mae, label='Training Loss (MAE)')
    plt.plot(range(1, num_epochs + 1), test_losses_mae, label='Test Loss (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MAE Learning Curves')
    plt.legend()

    # Add model information as text on the right side
    plt.text(1.02, 0.5, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    # Adjust layout to fit the text and plot within the image
    plt.subplots_adjust(right=0.7)  # Adjust the space on the right for the text
    plt.margins(0.2)  # Add some margin around the plot

    # Generate a unique filename with timestamp
    filename_mae = f'plots/mae_learning_curves_{timestamp}.png'
    plt.savefig(filename_mae)
    plt.show()

def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss_mse = 0.0
    running_loss_mae = 0.0
    running_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_mse = criterion(outputs, labels.float().unsqueeze(1))
            loss_mae = torch.mean(torch.abs(outputs.squeeze(1) - labels))
            absolute_errors = torch.abs(outputs.squeeze(1) - labels)
            running_loss_mse += loss_mse.item() * inputs.size(0)
            running_loss_mae += loss_mae.item() * inputs.size(0)
            running_mae += torch.sum(absolute_errors).item()
            total_samples += labels.size(0)

    test_loss_mse = running_loss_mse / total_samples
    test_loss_mae = running_loss_mae / total_samples
    test_mae = running_mae / total_samples
    return test_loss_mse, test_loss_mae, test_mae




file_path = FILE_PATH
skip_rows = SKIP_ROWS
num_rows = NUM_ROWS
batch_size = BATCH_SIZE
num_epochs = NUM_EPOCHS
lr = LEARNING_RATE


class ChessDataset(Dataset):
    def __init__(self, data_frame, batch_size):
        self.data_frame = data_frame
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data_frame) // self.batch_size

    def __getitem__(self, index):
        if isinstance(index, slice):
            start_idx, end_idx = index.start, index.stop
        else:
            start_idx = index * self.batch_size
            end_idx = (index + 1) * self.batch_size

        batch_data = self.data_frame.iloc[start_idx:end_idx]
        fens = torch.from_numpy(np.array([*map(fen_to_bit_vector, batch_data["FEN"])], dtype=np.float32))

        evals = batch_data["Evaluation"]
        evals = torch.Tensor(np.array(evals))

        return fens, evals


def plot_actual_vs_predicted(model, data_loader, device, is_training=True, percentage=0.1):
    model.eval()
    actual_values = []
    predicted_values = []

    # Calculate the number of samples to plot based on the percentage
    num_samples = int(len(data_loader.dataset) * percentage)

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            actual_values.extend(labels.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy().flatten())

            # Break the loop if we have reached the desired number of samples
            if len(actual_values) >= num_samples:
                break

    dataset_type = "Training" if is_training else "Testing"
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual', marker='o', linestyle='', color='b')
    plt.plot(predicted_values, label='Predicted', marker='x', linestyle='', color='r')
    plt.xlabel('Data Index')
    plt.ylabel('Evaluation')
    plt.title(f'Actual vs Predicted ({dataset_type} Data - {percentage * 100}% of Dataset)')
    plt.legend()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    chess_data = import_chess_data(file_path, skip_rows, num_rows)
    fen_data, evaluation_data = preprocess_data2(chess_data)



    print("FEN Data:")
    print(fen_data.head())
    print("\nEvaluation Data:")
    print(evaluation_data.head())

    normalized_evaluation_data = normalize_evaluation(evaluation_data)
    print("Normalized Evaluation Data:")
    print(normalized_evaluation_data.head())

    chess_data['FEN'] = fen_data
    chess_data['Evaluation'] = normalized_evaluation_data
    print(chess_data.columns)
    # Convert FEN data to bit vectors

    # Split data into training and testing sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(chess_data))
    test_size = int((1-train_ratio)* len(chess_data))
    train_data, test_data = chess_data[:train_size], chess_data[train_size:]

    train_dataset = ChessDataset(train_data, batch_size=batch_size)
    test_dataset = ChessDataset(test_data, batch_size=batch_size)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)


    # Initialize model, loss function, and optimizer
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    # Train the model
    train_model(model, criterion, optimizer, train_loader, test_loader, device, num_epochs=num_epochs, batch_size=batch_size, train_size=train_size,test_size=test_size)
    torch.save(model, 'chessModel.pth')

    end_time = time.time()  # Record end time

    elapsed_time_seconds = end_time - start_time  # Calculate elapsed time
    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_time = "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))
    print(f"Elapsed time: {elapsed_time} seconds")

    plot_actual_vs_predicted(model, train_loader, device, is_training=True)
    plot_actual_vs_predicted(model, test_loader, device, is_training=False)





if __name__ == "__main__":
    main()

