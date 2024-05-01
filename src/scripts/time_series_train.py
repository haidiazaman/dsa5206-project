import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..model.time_series_model import *
from ..dataloader.time_series_dataset import *
import matplotlib.pyplot as plt


def preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name):

    train_size,val_size,test_size = eval(train_val_test_split)

    #########################
    # DATASET PREPROCESSING
    #########################
    data = pd.read_csv(csvPath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data = data[[date_col_name,value_col_name]]
    data[date_col_name] = pd.to_datetime(data[date_col_name])
    plt.plot(data[date_col_name], data[value_col_name])
    plt.show()


    shifted_df = prepare_dataframe_for_lstm(data, lookback, date_col_name, value_col_name)
    shifted_df

    # format X and y from df and scale it
    shifted_df_as_np = shifted_df.to_numpy()
    print(shifted_df_as_np.shape)

    # normalise the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]
    print(X.shape, y.shape)
    X = dc(np.flip(X, axis=1))

    # train test split
    X_train = X[:int(len(X) * train_size)]
    X_val = X[int(len(X) * train_size):int(len(X) * train_size)+int(len(X) * val_size)]

    y_train = y[:int(len(y) * train_size)]
    y_val = y[int(len(y) * train_size):int(len(y) * train_size)+int(len(y) * val_size)]
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    X_train = X_train.reshape((-1, lookback, 1))
    X_val = X_val.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    X_train = torch.tensor(X_train).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_val = torch.tensor(y_val).float()
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True) # set all shuffle=False since its sequential data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # print to check
    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    return train_loader,val_loader
        
    
def train_rnn(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name,epochs):

    train_loader,val_loader = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    #########################
    # DEFINE MODEL AND TRAIN
    #########################
    # Train the model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = RNN(
        input_size=1,
        hidden_size=4,
        num_stacked_layers=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    early_stop_count = 0
    min_val_loss = float('inf')

    train_losses=[]
    val_losses=[]


    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'epoch {epoch}')
        model.train()

        train_running_loss = []
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_running_loss)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss.append(loss.item())

        val_loss = np.mean(val_running_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'rnn.pt')
            print(f'model epoch {epoch} saved as rnn.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 10:
            print("Early stopping!")
            break

        time_taken = round(time.time()-epoch_start_time,1)
        print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time_taken: {time_taken}")

        
def train_lstm(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name,epochs):

    train_loader,val_loader = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    #########################
    # DEFINE MODEL AND TRAIN
    #########################
    
    # Train the model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = LSTM(
        input_size=1,
        hidden_size=4,
        num_stacked_layers=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    early_stop_count = 0
    min_val_loss = float('inf')

    train_losses=[]
    val_losses=[]


    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'epoch {epoch}')
        model.train()

        train_running_loss = []
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_running_loss)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss.append(loss.item())

        val_loss = np.mean(val_running_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'lstm.pt')
            print(f'model epoch {epoch} saved as lstm.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 10:
            print("Early stopping!")
            break

        time_taken = round(time.time()-epoch_start_time,1)
        print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time_taken: {time_taken}")
    
def train_gru(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name,epochs):

    train_loader,val_loader = preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_name)

    #########################
    # DEFINE MODEL AND TRAIN
    #########################
    
    # Train the model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    model = GRU(
        input_size=1,
        hidden_size=4,
        num_stacked_layers=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    early_stop_count = 0
    min_val_loss = float('inf')

    train_losses=[]
    val_losses=[]


    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'epoch {epoch}')
        model.train()

        train_running_loss = []
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            train_running_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_loss = np.mean(train_running_loss)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_running_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss.append(loss.item())

        val_loss = np.mean(val_running_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'gru.pt')
            print(f'model epoch {epoch} saved as gru.pt')
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= 10:
            print("Early stopping!")
            break

        time_taken = round(time.time()-epoch_start_time,1)
        print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, time_taken: {time_taken}")
    
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='pretrain', choices=['pretrain','finetune','train'], help='train is only for rnn,lstm,gru, finetune and pretrain is for the transformer architectures')
    parser.add_argument('--model_type', default='mae', choices=['mae','no mae','rnn','lstm','gru'])
    parser.add_argument('--csvPath', default='ETTh1.csv', help='the directory of training data')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for training')
    parser.add_argument('--train_val_test_split', default='[0.7,0.2,0.1]', help='specify train-val-test split here, pass a string of list of floats')
    parser.add_argument('--lookback', type=int, default=100, help='indicate intended lookback period aka lags window')
    parser.add_argument('--date_col_name', default='date', help='specify date col name in data')
    parser.add_argument('--value_col_name', default='OT', help='specify value col name in data')
    parser.add_argument('--masking_ratio', type=float, default=0.7, help='specify masking ratio, only for pretraining mae')
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs for training')
    
    opt = parser.parse_args()

    if opt.phase == 'pretrain' and opt.model_type == 'mae':
        pretrain_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.masking_ratio,opt.epochs)

    elif opt.phase == 'finetune' and opt.model_type == 'mae':
        finetune_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.masking_ratio,opt.epochs)
        
    elif opt.phase == 'pretrain' and opt.model_type == 'no mae':
        pretrain_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)

    elif opt.phase == 'finetune' and opt.model_type == 'no mae':
        finetune_no_mae(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)
        
    elif opt.phase == 'train' and opt.model_type == 'rnn':
        train_rnn(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)
        
    elif opt.phase == 'train' and opt.model_type == 'lstm':
        train_lstm(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)
        
    elif opt.phase == 'train' and opt.model_type == 'gru':
        train_gru(opt.csvPath,opt.batch_size,opt.train_val_test_split,opt.lookback,opt.date_col_name,opt.value_col_name,opt.epochs)