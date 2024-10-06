import keras
import pandas as pd
import numpy as np
import statistics as st
import math
from keras import layers, regularizers
import sklearn.model_selection
from sklearn.linear_model import Ridge, LinearRegression, Lasso
import tensorflow as tf
import matplotlib.pyplot as plt
import os



# CUSTOM LAYERS
# Tied decoder layer with its own bias, taken from https://medium.com/@sahoo.puspanjali58/a-beginners-guide-to-build-stacked-autoencoder-and-tying-weights-with-it-9daee61eab2b
class DenseTranspose(keras.layers.Layer):
  def __init__(self, dense, activation=None, **kwargs):
      self.dense = dense
      self.activation = keras.activations.get(activation)
      super().__init__(**kwargs)
  def build(self, batch_input_shape):
      self.biases = self.add_weight(name="bias",    initializer="zeros",shape=[self.dense.input_shape[-1]])
      super().build(batch_input_shape)
  def call(self, inputs):
      z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
      return self.activation(z + self.biases)


# Load data
data_index_compression_df = pd.read_excel('./Index Compression Data-20230524/Data Index Compression.xlsx', sheet_name = 3)
# Interpolating the insisde area does nothing since we don't have NaN values between existing values. Anyway, we leave it for completeness. Also dropnaa
data_index_compression_df = data_index_compression_df.drop(columns = ['Dates']).interpolate(method='linear', limit_area = "inside").dropna()

list(data_index_compression_df.columns.values)

n_timestamps, n_stocks = data_index_compression_df.shape

dates = pd.read_excel("./Index Compression Data-20230524/Data Index Compression.xlsx", sheet_name = 3).loc[:n_timestamps-2, "Dates"]

# convert prices to numpy array and split rain and test
prices_npa = data_index_compression_df.to_numpy()
stocks_prices_train_npa, stocks_prices_test_npa = sklearn.model_selection.train_test_split(prices_npa, train_size = 0.875, shuffle = False)

def standardize(arr):
    return (arr - st.mean(arr)) / st.stdev(arr)

# Calculate returns, and center them
def get_standardized_returns_from_prices(arr, standardize_b=True):
    today = arr[1:]
    yesterday = arr[:-1]

    _returns = (today - yesterday) / yesterday
    if standardize_b:
        return standardize(_returns) #(_returns - st.mean(_returns)) / st.stdev(_returns)
    else:
        return _returns


returns_npa = np.zeros(shape = prices_npa.shape - np.array([1,0]))
returns_train_npa = np.zeros(shape = stocks_prices_train_npa.shape - np.array([1,0]))
returns_test_npa = np.zeros(shape = stocks_prices_test_npa.shape - np.array([1,0]))
for (i,col) in enumerate(prices_npa.T):
    returns_npa[:,i] = get_standardized_returns_from_prices(col)

for (i,col) in enumerate(stocks_prices_train_npa.T):
    returns_train_npa[:,i] = get_standardized_returns_from_prices(col)

for (i,col) in enumerate(stocks_prices_test_npa.T):
    returns_test_npa[:,i] = get_standardized_returns_from_prices(col)


""" returns_train_df = pd.DataFrame(columns = list(data_index_compression_df.columns.values))
returns_test_df = pd.DataFrame(columns = list(data_index_compression_df.columns.values))
for col_name in list(data_index_compression_df.columns.values):
    returns_train_df[col_name] = get_standardized_returns_from_prices(data_index_compression_df[col_name].to_numpy())
 """


def single_hidden_layer(input_len, encoding_dim, encoded_activation = "relu", decoded_activation = "linear", optimizer = "adam", loss = "mean_squared_error"):

    # This is the size of our encoded representations
    encoding_dim = 5
    # SINGLE-HIDDEN-LAYER AUTOENCODER WITH UNTIED WEIGHTS
    # input layer
    input_layer = keras.Input(shape=(n_stocks,))
    # latentr representation
    encoded = layers.Dense(encoding_dim, activation='selu')(input_layer)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(n_stocks, activation='linear')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_layer, decoded)
    # Save initialization weights before training, in order to later be able to reset the model should you need it.
    autoencoder.save_weights('model.h5')
    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder

shl_ae = single_hidden_layer(n_stocks, 5)

# fit
""" shl_ae.fit(returns_train_npa,returns_train_npa,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(returns_test_npa, returns_test_npa)) """

# reset the model
# autoencoder.load_weights('model.h5')

# SINGLE-HIDDEN-LAYER AUTOENCODER WITH TIED WEIGHTS
def single_hidden_layer_tied_weights(input_len, encoding_dim, encoded_activation = "relu", decoded_activation = "linear", optimizer = "adam", loss = "mean_squared_error") :
    # input_layer = keras.Input(shape=input_shape) # layers.InputLayer(input_shape=input_shape) 
    dense_1 = layers.Dense(encoding_dim, activation = encoded_activation)
    latent_representation = keras.models.Sequential([
        layers.InputLayer((input_len,)) ,
        dense_1
    ])

    decoded_representation =  keras.models.Sequential([
        DenseTranspose(dense_1, activation = decoded_activation),
        layers.Reshape([input_len,1])])

    autoencoder = keras.models.Sequential([latent_representation, decoded_representation])
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return  autoencoder

shl_tw_ae = single_hidden_layer_tied_weights(n_stocks, 5)

""" 
shl_tw_ae.fit(returns_train_npa,returns_train_npa,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(returns_test_npa, returns_test_npa)) """





# SINGLE-HIDDEN-LAYER REGULARIZED AUTOENCODER WITH TIED WEIGHTS
def single_hidden_layer_tied_weights_regularized(input_len, encoded_activation = "selu", decoded_activation = "linear", optimizer = "adam", loss = "mean_squared_error") :
    # input_layer = keras.Input(shape=input_shape) # layers.InputLayer(input_shape=input_shape) 
    dense_1 = layers.Dense(input_len, activation = encoded_activation, activity_regularizer = regularizers.l1(10e-5))
    latent_representation = keras.models.Sequential([
        layers.InputLayer((input_len,)) ,
        dense_1
    ])

    decoded_representation =  keras.models.Sequential([
        DenseTranspose(dense_1, activation = decoded_activation),
        layers.Reshape([input_len,1])])

    autoencoder = keras.models.Sequential([latent_representation, decoded_representation])
    autoencoder.compile(optimizer=optimizer, loss=loss)
    return  autoencoder


shl_tw_reg_ae = single_hidden_layer_tied_weights_regularized(n_stocks)


shl_tw_reg_ae.fit(returns_train_npa,returns_train_npa,
                epochs=250,
                batch_size=256,
                shuffle=False,
                validation_data=(returns_test_npa, returns_test_npa))


# COMMUNALITY
# It is calculated on the training set!

def get_communalities(measured_returns, predicted_returns):
    return np.sum((predicted_returns - measured_returns)**2, axis = 0)

reconstructed_returns_npa = shl_tw_reg_ae.predict(returns_npa).squeeze()

reconstructed_returns_train = shl_tw_reg_ae.predict(returns_train_npa).squeeze()
reconstructed_returns_train = reconstructed_returns_train

reconstructed_returns_test_npa = shl_tw_reg_ae.predict(returns_test_npa).squeeze()


# reconstructed_returns_test = shl_tw_reg_ae.predict(returns_test_npa)
# reconstructed_returns_test = reconstructed_returns_test.squeeze()

st.mean(reconstructed_returns_train[:,3])
st.stdev(reconstructed_returns_train[:,4])

communalities = get_communalities(returns_train_npa, np.apply_along_axis(standardize, 0, reconstructed_returns_train))
communalites_descending_sortperm = np.flip(np.argsort(communalities))


# I made the proportions with Zhang2020. I took as many least communals as possible, since the minium was 1, but we already have only 2 top communals...we may come up with our own proportions later
n_top_communal_to_keep   = 10 #math.ceil(40*(10/300))
n_least_communal_to_keep = 5 # math.ceil(40*(80/300) - n_top_communal_to_keep)

top_communals_prices_train = stocks_prices_train_npa[:,communalites_descending_sortperm[:n_top_communal_to_keep]]
top_communals_prices_test   = stocks_prices_test_npa[:,communalites_descending_sortperm[:n_top_communal_to_keep]]

top_communals_returns_train   = returns_train_npa[:,communalites_descending_sortperm[:n_top_communal_to_keep]]
top_communals_returns_test   = returns_test_npa[:,communalites_descending_sortperm[:n_top_communal_to_keep]]

least_communals_prices_train = stocks_prices_train_npa[:,communalites_descending_sortperm[-n_least_communal_to_keep:]]
least_communals_prices_test = stocks_prices_test_npa[:,communalites_descending_sortperm[-n_least_communal_to_keep:]]

least_communals_returns_train = returns_train_npa[:,communalites_descending_sortperm[-n_least_communal_to_keep:]]
least_communals_returns_test = returns_test_npa[:,communalites_descending_sortperm[-n_least_communal_to_keep:]]

selected_stocks_prices_train = np.c_[top_communals_prices_train,least_communals_prices_train]
selected_stocks_prices_test = np.c_[top_communals_prices_test,least_communals_prices_test]

selected_stocks_returns_train = np.c_[top_communals_returns_train,least_communals_returns_train]
selected_stocks_returns_test = np.c_[top_communals_returns_test,least_communals_returns_test]

selected_stocks_prices = np.append(selected_stocks_prices_train, selected_stocks_prices_test, axis=0)

st.mean(selected_stocks_returns_train[:,3])
st.stdev(selected_stocks_returns_train[:,4])

st.mean(selected_stocks_returns_test[:,3])
st.stdev(selected_stocks_returns_test[:,4])

# Function that takes the first price of a stock and its timestep-on-timestep interest, and returns the time series of prices
def apply_variable_comulative_interest(first_train_price, percentual_normalized_interests_time_series):
    terms = 1+percentual_normalized_interests_time_series/100
    cumulated_interests = np.cumprod(terms)
    return first_train_price*cumulated_interests


# INDEX TRACKING MODEL

# Index returns time series
## Load data
ftsemib_prices_xlsx = pd.read_excel("./Index Compression Data-20230524/Data Index Compression.xlsx", sheet_name = 2)
# Remove dates column and  truncate the dataset so that its elements correspond to those in data_index_compression_df timestamp-wise
ftsemib_prices_xlsx = ftsemib_prices_xlsx.drop(columns = ['Dates'])[:n_timestamps]

# Compute (normalized) returns

ftsemib_prices_npa = ftsemib_prices_xlsx.loc[:,"STH3 Index"].to_numpy()
ftsemib_prices_train_npa = ftsemib_prices_npa[:returns_train_npa.shape[0]+1]
ftsemib_prices_test_npa = ftsemib_prices_npa[returns_train_npa.shape[0]+1:]
ftsemib_normalized_returns = get_standardized_returns_from_prices(ftsemib_prices_npa)
ftsemib_normalized_returns_train = get_standardized_returns_from_prices(ftsemib_prices_xlsx.loc[:returns_train_npa.shape[0], "STH3 Index"].to_numpy())
ftsemib_normalized_returns_test = get_standardized_returns_from_prices(ftsemib_prices_xlsx.loc[(returns_train_npa.shape[0]+1):, "STH3 Index"].to_numpy())

st.mean(ftsemib_normalized_returns_train)
st.stdev(ftsemib_normalized_returns_train)

if False:
    # Plot the best reconstructed stock price together with its real price. It doesn't look very good, possibly for two reasons:
    # 1. The interests (returns) are normalized, so they cannot be used as they are below inside `apply_variable_comulative_interest`;
    # 2. The autoencoder still performs very badly
    stock_idx = communalites_descending_sortperm[0]
    first_train_price = data_index_compression_df.iloc[0, stock_idx]
    first_test_price = data_index_compression_df.iloc[returns_train_npa.shape[0], stock_idx]
    stocks_names = list(data_index_compression_df.columns.values)
    plt.plot(range(returns_train_npa.shape[0]), apply_variable_comulative_interest(first_train_price,returns_train_npa[:, stock_idx] ), label = "stock returns", linewidth=1.0 ) # dates[:returns_train_npa.shape[0]]
    plt.plot(range(returns_train_npa.shape[0]), apply_variable_comulative_interest(first_train_price,standardize(reconstructed_returns_train[:, stock_idx ])), label = "reconstructed stock returns", linewidth=1.0 )  #dates[:returns_train_npa.shape[0]]
    plt.title(stocks_names[stock_idx])
    plt.xlabel("timestamp")
    plt.ylabel("price (standardized returns)")
    plt.legend()
    plt.show()

    plt.savefig('./saved_figures/'+stocks_names[stock_idx]+"_reconstruction.png")


    plt.plot(range(returns_test_npa.shape[0]), apply_variable_comulative_interest(first_test_price,returns_test_npa[:, stock_idx] ), label = "stock returns", linewidth=1.0 ) # dates[:returns_train_npa.shape[0]]
    plt.plot(range(returns_test_npa.shape[0]), apply_variable_comulative_interest(first_test_price,standardize(reconstructed_returns_test_npa[:, stock_idx ])), label = "reconstructed stock returns", linewidth=1.0 )  #dates[:returns_train_npa.shape[0]]
    plt.title(stocks_names[stock_idx])
    plt.xlabel("timestamp")
    plt.ylabel("price (standardized returns)")
    plt.legend()
    plt.show()

    plt.savefig('./saved_figures/'+stocks_names[stock_idx]+"_reconstruction.png")


    plt.plot(range(returns_npa.shape[0]), apply_variable_comulative_interest(first_train_price,returns_npa[:, stock_idx] ), label = "stock returns", linewidth=1.0 ) # dates[:returns_train_npa.shape[0]]
    plt.plot(range(returns_npa.shape[0]), apply_variable_comulative_interest(first_train_price,standardize(reconstructed_returns_npa[:, stock_idx ])), label = "reconstructed stock returns", linewidth=1.0 )  #dates[:returns_train_npa.shape[0]]
    plt.title(stocks_names[stock_idx])
    plt.xlabel("timestamp")
    plt.ylabel("price (standardized returns)")
    plt.axvline(len(ftsemib_prices_train_npa), color='r')
    plt.legend()
    plt.show()

    plt.savefig('./saved_figures/'+stocks_names[stock_idx]+"_reconstruction.png")

    plt.plot(range(returns_test_npa.shape[0]), apply_variable_comulative_interest(first_train_price,returns_test_npa[:, stock_idx] ), "bo")
    plt.plot(range(returns_test_npa.shape[0]), apply_variable_comulative_interest(first_train_price,standardize(reconstructed_returns_test_npa[:,stock_idx])) )  #dates[:returns_npa.shape[0]]
    plt.show() 

    plt.plot(range(returns_npa.shape[0]), apply_variable_comulative_interest(first_train_price,returns_npa[:, stock_idx] ), "bo")
    plt.plot(range(returns_npa.shape[0]), apply_variable_comulative_interest(first_train_price,standardize(reconstructed_returns_npa[:,stock_idx])) )  #dates[:returns_npa.shape[0]]
    plt.show() 


    plt.plot(dates[:returns_train_npa.shape[0]], data_index_compression_df.iloc[:returns_train_npa.shape[0], stock_idx])
    plt.plot(dates[:returns_train_npa.shape[0]], apply_variable_comulative_interest(data_index_compression_df.iloc[0, stock_idx], reconstructed_returns_train[:,stock_idx ] ) ) #itm_ridge.coef_[0]*
    plt.show()

    st.mean(reconstructed_returns_train[:,stock_idx ])
    st.stdev(reconstructed_returns_train[:,stock_idx ])
    plt.show()

# Perform regression of the INDEX TRACKING MODEL
itm_ridge = Ridge(alpha = 1, positive = True, fit_intercept=  False)
itm_ridge.fit(selected_stocks_prices_train, ftsemib_prices_train_npa) # shares_prices_npa[:3166, communalites_descending_sortperm[:n_top_communal_to_keep]]
itm_ridge.score(selected_stocks_prices_train, ftsemib_prices_train_npa)

index_prices_from_selected_stocks_npa = []
for i in range(len(selected_stocks_prices)):
    weighted_sum = selected_stocks_prices[i] * itm_ridge.coef_ # weights_npa itm_ridge.coef_
    index_prices_from_selected_stocks_npa.append(sum(weighted_sum))

plt.plot(range(3619), index_prices_from_selected_stocks_npa, label='weighted sums') # index_price_npa[:3618]
plt.plot(range(3619), ftsemib_prices_npa, label='actual index price')
plt.axvline(len(ftsemib_prices_train_npa), color='r')
plt.legend()
plt.show()

itm_ridge.score(selected_stocks_prices_test, ftsemib_prices_test_npa)

itm_ridge.coef_

itm_ridge.score()

itm_ridge.get_params()


itm_returns_ridge = Ridge(alpha = 1, positive = False, fit_intercept=  False)
itm_returns_ridge.fit(selected_stocks_returns_train, returns_train_npa) # shares_prices_npa[:3166, communalites_descending_sortperm[:n_top_communal_to_keep]]
itm_returns_ridge.score(selected_stocks_prices_train, ftsemib_prices_train_npa)

# Lasso regressor
itm_lasso = Lasso(alpha = 0.01, positive = True, fit_intercept=  False)
itm_lasso.fit(selected_stocks_prices_train, ftsemib_prices_train_npa)
itm_lasso.score(selected_stocks_prices_train, ftsemib_prices_train_npa)

index_prices_from_selected_stocks_npa = []
for i in range(len(selected_stocks_prices)):
    weighted_sum = selected_stocks_prices[i] * itm_lasso.coef_ # weights_npa itm_ridge.coef_
    index_prices_from_selected_stocks_npa.append(sum(weighted_sum))

plt.plot(range(3619), index_prices_from_selected_stocks_npa, label='weighted sums') # index_price_npa[:3618]
plt.plot(range(3619), ftsemib_prices_npa, label='actual index price')
plt.axvline(len(ftsemib_prices_train_npa), color='r')
plt.legend()
plt.show()



# ATE
def ate(R_i, R_p):  
    # len(R_i)  total number of out-of-sample trading days
    return np.sqrt((1/len(R_i))*np.sum((R_i - R_p)**2))

ate(ftsemib_normalized_returns_test, np.sum(selected_stocks_returns_test*itm_ridge.coef_, axis=1) )
ate(ftsemib_prices_test_npa, np.sum(selected_stocks_prices_test*itm_ridge.coef_, axis=1) )

# MARKET VALUE RANKING
ftsemib_weights_npa = pd.read_excel("./Index Compression Data-20230524/Data Index Compression.xlsx", sheet_name = 4).loc[:,'Shares per Basket'].to_numpy()
ftsemib_weights_sortperm = np.flip(np.argsort(ftsemib_weights_npa))

n = n_top_communal_to_keep + n_least_communal_to_keep

returns_train_selected_market_value = returns_train_npa[:,ftsemib_weights_sortperm[:n]]
returns_test_selected_market_value = returns_test_npa[:,ftsemib_weights_sortperm[:n]]

itm_ridge_market_value = Ridge(alpha = 0.01, positive = True, fit_intercept=  False)
itm_ridge_market_value.fit(returns_train_selected_market_value, ftsemib_normalized_returns_train)
itm_ridge_market_value.score(returns_train_selected_market_value, ftsemib_normalized_returns_train)

itm_ridge_market_value.coef_

ate( ftsemib_normalized_returns_test, np.sum(returns_test_selected_market_value*itm_ridge_market_value.coef_, axis=1) )

plt.plot(np.cumsum(ftsemib_weights_npa[(ftsemib_weights_sortperm)]))
plt.show()


# :::::WARNING::::: 
# TESTING: the relation between index prices and weighted sum of stocks
index_price = pd.read_excel('./Index Compression Data-20230524/Data Index Compression.xlsx', sheet_name = 2)
shares_prices_df = pd.read_excel('./Index Compression Data-20230524/Data Index Compression.xlsx', sheet_name = 3).drop(columns = ['Dates']).interpolate(method='linear', limit_area = "inside").dropna()
shares_per_basket_df = pd.read_excel('./Index Compression Data-20230524/Data Index Compression.xlsx', sheet_name = 4)

weights_npa = shares_per_basket_df['Shares per Basket'].to_numpy()
shares_prices_npa = shares_prices_df.values[1:]
index_price_npa = index_price['STH3 Index'].to_numpy()[:3618]

dates = index_price['Dates'][:3618]

# RIDGE: INDEX PRICE - REBUILT PRICE (non standardized)
index_prices_from_weighted_stocks_npa = []
for i in range(len(shares_prices_npa)):
    weighted_sum = shares_prices_npa[i] * weights_npa # weights_npa itm_ridge.coef_
    index_prices_from_weighted_stocks_npa.append(sum(weighted_sum))

index_prices_from_weighted_stocks_npa = np.array(index_prices_from_weighted_stocks_npa)

plt.plot(dates, index_prices_from_weighted_stocks_npa, label='weighted sums') # index_price_npa[:3618]
plt.plot(dates, index_price_npa, label='actual index price')
plt.legend()
plt.show()

itm_ridge = Ridge(alpha = 1, positive = True, fit_intercept=  False)
itm_ridge.fit(shares_prices_npa, index_price_npa)
itm_ridge.score(shares_prices_npa, index_price_npa)

itm_ridge.coef_
itm_ridge.score()
itm_ridge.get_params()
itm_ridge.coef_ - weights_npa

# RIDGE: INDEX RETURN - REBUILT RETURNS (non standardized)
index_returns_npa = [(index_price_npa[i] - index_price_npa[i-1])/index_price_npa[i-1] for i in range(1,len(index_price_npa))]
# index_returns_from_weighted_stocks_npa = [(index_prices_from_weighted_stocks_npa[i] - index_prices_from_weighted_stocks_npa[i-1])/index_prices_from_weighted_stocks_npa[i-1] for i in range(1,len(index_prices_from_weighted_stocks_npa))]
index_returns_from_weighted_stocks_npa = np.zeros(shares_prices_npa.shape - np.array([1,0]))

for (i,share_prices) in enumerate(shares_prices_npa.T):
    index_returns_from_weighted_stocks_npa[:,i] = get_standardized_returns_from_prices(share_prices, False)

returns_ridge = Ridge(alpha = 1, positive = True, fit_intercept=  False)
returns_ridge.fit(index_returns_from_weighted_stocks_npa, index_returns_npa)
returns_ridge.score(index_returns_from_weighted_stocks_npa, index_returns_npa)

returns_ridge.coef_
returns_ridge.score()
returns_ridge.get_params()
returns_ridge.coef_ - weights_npa

# MULTI STOCK PLOT 
ordered_stocks_isin = shares_per_basket_df.iloc[communalites_descending_sortperm]['ISIN'].to_list()
selected_stocks_isin = ordered_stocks_isin[0:n_top_communal_to_keep]
selected_stocks_isin.extend(ordered_stocks_isin[-n_least_communal_to_keep:])

fig, axs = plt.subplots(3,5)

# axs[0,0].plot(range(3619), selected_stocks_prices.T[0])
# axs[0,1].plot(range(3619), selected_stocks_prices.T[1])
# axs[0,2].plot(range(3619), selected_stocks_prices.T[2])
# axs[0,3].plot(range(3619), selected_stocks_prices.T[3])
# axs[0,4].plot(range(3619), selected_stocks_prices.T[4])

# axs[1,0].plot(range(3619), selected_stocks_prices.T[5])
# axs[1,1].plot(range(3619), selected_stocks_prices.T[6])
# axs[1,2].plot(range(3619), selected_stocks_prices.T[7])
# axs[1,3].plot(range(3619), selected_stocks_prices.T[8])
# axs[1,4].plot(range(3619), selected_stocks_prices.T[9])

# axs[2,0].plot(range(3619), selected_stocks_prices.T[10])
# axs[2,1].plot(range(3619), selected_stocks_prices.T[11])
# axs[2,2].plot(range(3619), selected_stocks_prices.T[12])
# axs[2,3].plot(range(3619), selected_stocks_prices.T[13])
# axs[2,4].plot(range(3619), selected_stocks_prices.T[14])

for i in range(1,4):
    for j in range(1,6):
        if i*j<n_least_communal_to_keep+n_top_communal_to_keep:
            axs[i-1,j-1].plot(range(3619), selected_stocks_prices.T[(i*j)-1])
            axs[i-1,j-1].axvline(len(selected_stocks_prices_train), color='r')
            axs[i-1,j-1].set_title(selected_stocks_isin[(i*j)-1])

plt.show(block=True)