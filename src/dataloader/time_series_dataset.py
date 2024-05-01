import pandas as pd
from torch.utils.data import Dataset
from copy import deepcopy as dc


# format dataset into appropriate format for numpy reshaping
def prepare_dataframe(df, n_steps, date_col_name, value_col_name):
    df = dc(df)
    df.set_index(date_col_name, inplace=True)
    for i in range(1, n_steps+1):
        df[f'{value_col_name}(t-{i})'] = df[value_col_name].shift(i)
    df.dropna(inplace=True)
    return df

# define dataloader class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    
def preprocess_data(csvPath,batch_size,train_val_test_split,lookback,date_col_name,value_col_names):
    
    # split dataset
    train_size,val_size,test_size = eval(train_val_test_split)

    # read csv and convert date col to datetime format
    data = pd.read_csv(csvPath)
    data[date_col_name] = pd.to_datetime(data[date_col_name])

    for value_col_name in value_col_names:
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