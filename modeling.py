import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
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

def plot_multistep(y, every=1, ax=None, palette_kwargs=None):
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

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

output_path = './Plot_Figures'
DPI_OUTPUT = 300
def plot_stepped_predictions(model, X_val, y_val, name, prediction_step = 3):
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
print(x_train.head())
print(y_train.head())

# Basic feed forward neural network    
num_features = x_train.shape[-1]
num_outputs = y_train.shape[-1]
print(f"Number of outputs: {num_outputs}, Number of inputs: {num_features}")

model = keras.Sequential(
    [
        layers.Dense(64, input_dim=num_features, kernel_initializer='he_uniform', activation='relu'),
        layers.Dense(64, kernel_initializer='he_uniform', activation='relu'),
	    layers.Dense(num_outputs, kernel_initializer='he_uniform')
    ]
)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train and Eval model
model.fit(x_train, y_train, batch_size=10)
evaluate(model, x_train, y_train)
evaluate(model, x_val, y_val)

# plot predictions
plot_stepped_predictions(model, x_val, y_val, '64-64-node-NN')