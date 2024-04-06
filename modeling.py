import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from math import ceil

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential

import random
from numpy.random import seed

# Set seeding
# seed_num = 7548
# seed(seed_num)
# tf.random.set_seed(seed_num)
# print('The seed is: ', seed_num)

plt.style.use('bmh')

def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        legend=False,
    )
    palette_kwargs_ = dict(palette='husl', n_colors=16, desat=None)
    if palette_kwargs is not None:
        palette_kwargs_.update(palette_kwargs)
    palette = sns.color_palette(**palette_kwargs_)
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_prop_cycle(plt.cycler('color', palette))
    for date, preds in y[::every].iterrows():
        preds.index = pd.period_range(start=date, periods=len(preds))
        preds.plot(ax=ax)
    return ax

output_path = './Plot_Figures'
DPI_OUTPUT = 300
def plot_stepped_predictions(model, X_val, y_val, name, prediction_step = 3):
    plot_params = dict(
        color="0.75",
        style=".-",
        markeredgecolor="0.25",
        markerfacecolor="0.25",
        legend=False,
    )
    
    pred = pd.DataFrame(
        model.predict(X_val),
        index = y_val.index, columns = y_val.columns
    )
    palette = dict(palette='husl', n_colors=64)
    ax2 = y_val['y_step_1'].plot(**plot_params)
    ax2 = plot_multistep(pred, ax=ax2, palette_kwargs=palette, every=prediction_step)
    _ = ax2.legend(['Value', 'Forecast'])

    plot_name = f'{name}_stepped_predictions'
    plt.title(plot_name)
    plt.tight_layout()
    plt.savefig(f'{output_path}/{plot_name}.png', dpi = DPI_OUTPUT)
    
def evaluate(model, X_val, y_val):
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1]) if len(y_val.shape) > 2 else y_val
    pred = model.predict(X_val)
    pred = pred.reshape(pred.shape[0], pred.shape[1]) if len(pred.shape) > 2 else pred

    mse = mean_squared_error(pred, y_val)
    mape = mean_absolute_percentage_error(pred, y_val)
    print('Result - MSE: ', mse, ' - MAPE: ', mape)
    
in_names = ['train/x_train_scaled', 'test/x_val_scaled', 'train/y_train', 'test/y_val']
x_train, x_val, y_train, y_val = [pd.read_csv(f'./data/{name}.csv', index_col = 'Date', parse_dates=True) for name in in_names]

model = RegressorChain(base_estimator= RandomForestRegressor())
model.fit(x_train, y_train)

evaluate(model, x_train, y_train)
evaluate(model, x_val, y_val)

plot_stepped_predictions(model, x_val, y_val, 'RanFor')

# Plot feature importance
cols = 2
rows = ceil(len(model.estimators_) / float(cols))
fig, axes = plt.subplots(rows, cols, figsize=(20,15))

for idx, estimator in enumerate(model.estimators_):

    org_features = list(model.feature_names_in_)
    prev_reg_names = [f'Prev regressor {i}' for i in range(idx)]
    feature_names = org_features + prev_reg_names
    
    ax = axes.flat[idx]
    ax.title.set_text(f'Feature Importance of model step {idx}')
    
    importance = estimator.feature_importances_
    
    df = pd.DataFrame(
        importance,
        index = feature_names, columns = ['imp']
    ).abs().sort_values(by='imp')
    df.plot.barh(ax=ax)
    
name = 'RanFor'
plot_name = f'{name}_feature_importance'
plt.title(plot_name)
plt.savefig(f'{output_path}/{plot_name}.png', dpi=DPI_OUTPUT)

# # Basic feed forward neural network    
# num_features = x_train.shape[-1]
# num_outputs = y_train.shape[-1]
# print(f"Number of outputs: {num_outputs}, Number of inputs: {num_features}")

# model = keras.Sequential(
#     [
#         layers.Dense(150, input_dim=num_features, kernel_initializer='he_normal', activation='relu'),
#         layers.Dense(200, kernel_initializer='he_normal', activation='relu'),
#         layers.Dense(150, kernel_initializer='he_normal', activation='relu'),
#         layers.Dense(200, kernel_initializer='he_normal', activation='relu'),
# 	    layers.Dense(num_outputs, kernel_initializer='he_normal')
#     ]
# )
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.summary()

# # Train and Eval model
# print(x_train.shape, y_train.shape)
# model.fit(x_train, y_train, batch_size=20)
# evaluate(model, x_train, y_train)
# evaluate(model, x_val, y_val)

# # plot predictions
# plot_stepped_predictions(model, x_val, y_val, '150-200-150-200-node-NN')