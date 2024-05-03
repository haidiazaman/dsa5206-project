# dsa5206-project
This repo consolidates the work done for my dsa5206 project. The project pdf has been added as well. 

# Project: Is Non-Linear Always Better?
In this project, I explore the effects of linear vs non-linear dimensionality reduction on downstream time series forecasting tasks.

The dataset used for this paper can be downloaded here: https://www.kaggle.com/datasets/mnassrib/jena-climate/data.

Context: Jena Climate is weather timeseries dataset recorded at the Weather Station of the Max Planck Institute for Biogeochemistry in Jena, Germany. Jena Climate dataset is made up of 14 different quantities (such air temperature, atmospheric pressure, humidity, wind direction, and so on) were recorded every 10 minutes, over several years. This dataset covers data from January 1st 2009 to December 31st 2016.

# Methods
The project involves 4 main sets of experiments. 
- Baseline experiments using the full set of 14 features followed by time series forecasting.
- Linear Principal Component Analysis (PCA n=2) followed by time series forecasting.
- Non-linear Autoencoder followed by time series forecasting.
- PCA n=3 followed by time series forecasting.
Note that in each sets of experiments, the resulting data from the dimensionality reduction method is used to train 3 different deep learning architectures: RNN, LSTM, GRU. The network structure of each cell can be seen here: ![alt text](https://github.com/haidiazaman/dsa5206-project/blob/main/imgs/Network-Structure-of-RNN-LSTM-and-GRU%20(1).png)

# Results of the 2nd sets of experiments: Linear PCA followed by forecasting
This is the plot for the first 100 time steps of the test dataset only

![alt text](https://github.com/haidiazaman/dsa5206-project/blob/main/imgs/expt2.png)

# Comparison results btwn PCA(n=2) vs PCA(n=3)
It can be seen here that PCA(n=3) produces a representation that is better than PCA(n=3) due to the additional principal component covering more of the total variance of the dataset, thus creating a reduced representation of the data that is closer to the original dataset than PCA(n=2)

![alt text](https://github.com/haidiazaman/dsa5206-project/blob/main/imgs/expt4.jpeg)
