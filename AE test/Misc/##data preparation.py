##data preparation
T = 2 # two day lags
HORIZON = 30 # predict 30 days ahead
# Step 1: get the train data from the correct data range
train = ts_data_load.copy()[ts_data_load.index < valid_st_data_load]
[['load']]
 
# Step 2: scale data to be in range (0, 1). 
scaler = MinMaxScaler()
train['load'] = scaler.fit_transform(train)
 
# Step 3: shift the dataframe to create the input samples
train_shifted = train.copy()
train_shifted['y_t+1'] = train_shifted['load'].shift(-1, freq='H')
for t in range(1, T+1):
    train_shifted[str(T-t)] = train_shifted['load'].shift(T-t, freq='H')
y_col = 'y_t+1'
X_cols = ['load_t-5',
             'load_t-4',
             'load_t-3',
             'load_t-2',
             'load_t-1',
             'load_t']
train_shifted.columns = ['load_original']+[y_col]+X_cols
 
# Step 4: discard missing values
train_shifted = train_shifted.dropna(how='any')
train_shifted.head(5)
# Step 5: transform this pandas dataframe into a numpy array
y_train = train_shifted[y_col].as_matrix()
X_train = train_shifted[X_cols].as_matrix()
#end of suervised learning format


# 3d tensor for LSTM
X_train = X_train.reshape(X_train.shape[0], T, 1)






from tscv import GapRollForward
cv = GapRollForward(min_train_size=3, gap_size=1, max_test_size=2)
for train, test in cv.split(range(10)):
    print("train:", train, "test:", test)
    
    
from sklearn import svm
from sklearn.model_selection import cross_val_score
from tscv import GapKFold

clf = svm.SVC(kernel='linear', C=1)
cv = GapKFold(n_splits=5, gap_before=5, gap_after=5)
scores = cross_val_score(clf, iris.data, iris.target, cv=cv)  
    
    
    
cv_rmse = pd.DataFrame.from_dict(cv_rmse, orient='index')
best_n, best_rmse = cv_rmse.mean(1).idxmin(), cv_rmse.mean(1).min()
cv_rmse = cv_rmse.stack().reset_index()
cv_rmse.columns =['n', 'fold', 'RMSE']    
    
    #pipeline note to copare must transofrm test set
    >>> from sklearn.pipeline import make_pipeline
>>> clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
>>> cross_val_score(clf, X, y, cv=cv)
    
    
    
    
# fit the model
my_rf = RandomForestRegressor()
my_rf.fit(X, co2_data.co2.values)

# predict on the same period
preds = my_rf.predict(X)



# fit the model
my_xgb = xgb.XGBRegressor()
my_xgb.fit(X, co2_data.co2.values)

# predict on the same period
preds = my_xgb.predict(X)



#example
pipe = Pipeline([('scaler', StandardScaler()), 
                 ('knn', KNeighborsRegressor())])

n_folds = 5
n_neighbors = tuple(range(5, 101, 5))

param_grid = {'knn__n_neighbors': n_neighbors}

estimator = GridSearchCV(estimator=pipe,
                         param_grid=param_grid,
                         cv=n_folds,
                         scoring=rmse_score,
#                          n_jobs=-1
                        )
estimator.fit(X=X, y=y)

cv_results = estimator.cv_results_
test_scores = pd.DataFrame({fold: cv_results[f'split{fold}_test_score'] for fold in range(n_folds)}, 
                           index=n_neighbors).stack().reset_index()
test_scores.columns = ['k', 'fold', 'RMSE']


Features:

#  Date time features
# ■ Lag features and window features
# Add quarter of the year
# add saturday
# add holidays


ts_data['hour'] = [ts_data.index[i].hour for i in range(len(ts_data))]
ts_data['month'] = [ts_data.index[i].month for i in range(len(ts_data))]
ts_data['dayofweek'] = [ts_data.index[i].day for i in range(len(ts_
data))]

# standardise the numeric features
# grid search optimise:
# 	cross validation train test split(walk froward)
# 		model training
# 		evaluation
	
 
 
 
 ########lightgbm
 # Split the data into training and test sets
train_data, test_data, train_target, test_target = train_test_split(
    df.drop(['target_variable'], axis=1), df['target_variable'], test_size=0.2, random_state=42)

# Define the LightGBM dataset format
train_dataset = lgb.Dataset(train_data, label=train_target)
test_dataset = lgb.Dataset(test_data, label=test_target)

# Set the LightGBM parameters for regression
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the LightGBM model
model = lgb.train(params, train_dataset, num_boost_round=1000, valid_sets=test_dataset, early_stopping_rounds=100)

# Make predictions on the test set
predictions = model.predict(test_data, num_iteration=model.best_iteration)

#####RF
rf_reg = RandomForestRegressor(n_estimators=100, 
                                max_depth=None, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                min_impurity_decrease=0.0, 
                                min_impurity_split=None, 
                                bootstrap=True, 
                                oob_score=False, 
                                n_jobs=-1, 
                                random_state=None, 
                                verbose=0, 
                                warm_start=False)
cv_score = cross_val_score(estimator=rf_reg,
                           X=X,
                           y=y,
                           scoring=ic,
                           cv=cv,
                           n_jobs=-1,
                           verbose=1)

param_grid = {'n_estimators': [50, 100, 250],
              'max_depth': [5, 15, None],
              'min_samples_leaf': [5, 25, 100]}

param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 10, 12, 15],
              'min_samples_leaf': [5, 25, 50, 100],
              'max_features': ['sqrt', 'auto']}

gridsearch_reg = GridSearchCV(estimator=rf_reg,
                      param_grid=param_grid,
                      scoring=ic,
                      n_jobs=-1,
                      cv=cv,
                      refit=True,
                      return_train_score=True,
                      verbose=1)	

gs_reg = gridsearch_reg
gridsearch_reg.fit(X=X, y=y)
# 	k cv score


# Starting at the beginning of the time series, the minimum number of samples in the
# window is used to train a model.
# 2. The model makes a prediction for the next time step.
# 3. The prediction is stored or evaluated against the known value.
# 4. The window is expanded to include the known value and the process is repeated (go to
# step 1.)



# ARIMA


models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsReg

names = []
kfold_results = []
test_results = []
train_results = []
for name, model in models:
    names.append(name)
    ## k-fold analysis:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    #converted mean squared error to positive. The lower the better
    cv_results = -1* cross_val_score(model, X_train, Y_train, cv=kfold, \
      scoring=scoring)
    kfold_results.append(cv_results)
    # Full Training period
    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)
    # Test results
    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)
	
	fig = pyplot.figure()
fig.suptitle('Algorithm Comparison: Kfold results')
ax = fig.add_subplot(111)
pyplot.boxplot(kfold_results)
ax.set_xticklabels(names)
fig.set_size_inches(15,8)
pyplot.show()




LSTM
seq_len = 2 #Length of the seq for the LSTM

Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
X_train_LSTM = np.zeros((X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]))
X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
    X_test_LSTM[:, i, :] = np.array(X)\
    [X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]

# LSTM Network
def create_LSTMmodel(learn_rate = 0.01, momentum=0):
        # create model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1],\
      X_train_LSTM.shape[2])))
    #More cells can be added if needed
    model.add(Dense(1))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer='adam')
    return model
LSTMModel = create_LSTMmodel(learn_rate = 0.01, momentum=0)
LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, \
  validation_data=(X_test_LSTM, Y_test_LSTM),\
  epochs=330, batch_size=72, verbose=0, shuffle=False)
  
  
  pyplot.plot(LSTMModel_fit.history['loss'], label='train', )
pyplot.plot(LSTMModel_fit.history['val_loss'], '--',label='test',)
pyplot.legend()
pyplot.show()

error_Training_LSTM = mean_squared_error(Y_train_LSTM,\
  LSTMModel.predict(X_train_LSTM))
predicted = LSTMModel.predict(X_test_LSTM)
error_Test_LSTM = mean_squared_error(Y_test,predicted)

test_results.append(error_Test_ARIMA)
test_results.append(error_Test_LSTM)

train_results.append(error_Training_ARIMA)
train_results.append(error_Training_LSTM)

names.append("ARIMA")
names.append("LSTM")






model = MLPRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, \
  cv=kfold)
grid_result = grid.fit(X_train, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))












import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import neptune
from sklearn.metrics import mean_squared_error

# specify the grid for the grid search of hyperparameter tuning
parameters={'max_depth': list(range(2, 20, 4)),
            'gamma': list(range(0, 10, 2)),
            'min_child_weight' : list(range(0, 10, 2)),
            'eta': [0.01,0.05, 0.1, 0.15,0.2,0.3,0.5]
    }

param_list = [(x, y, z, a) for x in parameters['max_depth'] for y in parameters['gamma'] for z in parameters['min_child_weight'] for a in parameters['eta']]


for params in param_list:

    mses = []

    run = neptune.init_run(
          project="YOU/YOUR_PROJECT",
          api_token="YOUR_API_TOKEN",
      )

    run['params'] = params

    my_kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in my_kfold.split(X_windows):


        X_train, X_test = X_windows[train_index], X_windows[test_index]
        y_train, y_test = np.array(y_data)[train_index], np.array(y_data)[test_index]

        xgb_model = xgb.XGBRegressor(max_depth=params[0],gamma=params[1], min_child_weight=params[2], eta=params[3])
        xgb_model.fit(X_train, y_train)
        preds = xgb_model.predict(X_test)

        mses.append(mean_squared_error(y_test, preds))

    average_mse = np.mean(mses)
    std_mse = np.std(mses)
    run['average_mse'] = average_mse
    run['std_mse'] = std_mse



import yfinance as yf
sp500_data = yf.download('^GSPC', start="1980-01-01", end="2021-11-21")
sp500_data = sp500_data[['Close']]
difs = (sp500_data.shift() - sp500_data) / sp500_data
difs = difs.dropna()
y = difs.Close.values
# create windows
X_data = []
y_data = []
for i in range(len(y) - 3*31):
    X_data.append(y[i:i+3*31])
    y_data.append(y[i+3*31])
X_windows = np.vstack(X_data)
# create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_windows, np.array(y_data), test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# build LSTM using tensorflow keras
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
import neptune
from sklearn.metrics import mean_squared_error
archi_list = [
              [tf.keras.layers.LSTM(32, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(64, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(128, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(128, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(32, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.LSTM(32, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],
              [tf.keras.layers.LSTM(64, return_sequences=True,  input_shape=(3*31,1)),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.LSTM(64, return_sequences=True),
               tf.keras.layers.Dense(units=1)
               ],

]

for archi in archi_list:
    run = neptune.init_run(
          project="YOU/YOUR_PROJECT",
          api_token="YOUR_API_TOKEN",
      )

    run['params'] = str(len(archi) - 1) + ' times ' + str(archi[0].units)
    run['Tags'] = 'lstm'


    lstm_model = tf.keras.models.Sequential(archi)
    lstm_model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanSquaredError()]
                      )
    history = lstm_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    run['last_mse'] = history.history['val_mean_squared_error'][-1]
    run.stop()





# Grid search

param_grid = dict(
        learning_rate=[.01, .1, .2],
        max_depth=list(range(3, 13, 3)),
        max_features=['sqrt', .8, 1],
        min_impurity_decrease=[0, .01],
        min_samples_split=[10, 50],
        n_estimators=[100, 300],
        subsample=[.8, 1])

all_params = list(product(*param_grid.values()))
print('# Models = :', len(all_params))

gs = GridSearchCV(gb_clf,
                  param_grid,
                  cv=cv,
                  scoring='roc_auc',
                  verbose=3,
                  n_jobs=-1,
                  return_train_score=True)
start = time()
gs.fit(X=X, y=y)
done = time()


lgb_train_params = ['learning_rate', 'num_leaves', 'feature_fraction', 'min_data_in_leaf']
scope_params = ['lookahead', 'train_length', 'test_length']
lgb_data = lgb.Dataset(data=data[features],
                       label=data[label],
                       categorical_feature=categoricals,
                       free_raw_data=False)


#gbm 
base_params = dict(boosting='gbdt',
                   objective='regression',
                   verbose=-1)
# constraints on structure (depth) of each tree
max_depths = [2, 3, 5, 7]
num_leaves_opts = [2 ** i for i in max_depths]
min_data_in_leaf_opts = [250, 500, 1000]

# weight of each new tree in the ensemble
learning_rate_ops = [.01, .1, .3]

# random feature selection
feature_fraction_opts = [.3, .6, .95]
param_names = ['learning_rate', 'num_leaves',
               'feature_fraction', 'min_data_in_leaf']
cv_params = list(product(learning_rate_ops,
                         num_leaves_opts,
                         feature_fraction_opts,
                         min_data_in_leaf_opts))
n_params = len(cv_params)
print(f'# Parameters: {n_params}')










#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)



##LSTM 2

layer_units, optimizer = 50, 'adam'
    cur_epochs = 15
    cur_batch_size = 20

    cur_LSTM_pars = {'units': layer_units,
                     'optimizer': optimizer,
                     'batch_size': cur_batch_size,
                     'epochs': cur_epochs
                     }

# Create an experiment and log the model in Neptune new version
npt_exp = neptune.init(
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject,
        name='LSTM',
        description='stock-prediction-machine-learning',
        tags=['stockprediction', 'LSTM','neptune'])
npt_exp['LSTMPars'] = cur_LSTM_pars
Next, we scale the input data for LSTM model regulation and split it into train and test sets.

# scale our dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[['Close']])
    scaled_data_train = scaled_data[:train.shape[0]]

# We use past 50 days’ stock prices for our training to predict the 51th day's closing price.
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)
A couple of notes:

# we use the StandardScaler, rather than the MinMaxScaler as you might have seen before. The reason is that stock prices are ever-changing, and there are no true min or max values. It doesn’t make sense to use the MinMaxScaler, although this choice probably won’t lead to disastrous results at the end of the day;
# stock price data in its raw format can’t be used in an LSTM model directly; we need to transform it using our pre-defined `extract_seqX_outcomeY` function. For instance, to predict the 51st price, this function creates input vectors of 50 data points prior and uses the 51st price as the outcome value.
# Moving on, let’s kick off the LSTM modeling process. Specifically, we’re building an LSTM with two hidden layers, and a ‘linear’ activation function upon the output. Also, this model is logged in Neptune.

### Build a LSTM model and log model summary to Neptune ###    
def Run_LSTM(X_train, layer_units=50, logNeptune=True, NeptuneProject=None):
    inp = Input(shape=(X_train.shape[1], 1))

    x = LSTM(units=layer_units, return_sequences=True)(inp)
    x = LSTM(units=layer_units)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)

    # Compile the LSTM neural net
    model.compile(loss = 'mean_squared_error', optimizer = 'adam')

    ## !!! log to Neptune, e.g., set NeptuneProject = npt_exp (new version)
    if logNeptune:
       model.summary(print_fn=lambda x: NeptuneProject['model_summary'].log(x))

    return model

model = Run_LSTM(X_train, layer_units=layer_units, logNeptune=True, NeptuneProject=npt_exp)

history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size,
                    verbose=1, validation_split=0.1, shuffle=True)
Model hyper-parameters and summary have been logged into Neptune.


# See in the app
# Once the training completes, we’ll test the model against our hold-out set. 

# predict stock prices using past window_size stock prices
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
    raw = data['Close'][len(data) - len(test) - window_size:].values
    raw = raw.reshape(-1,1)
    raw = scaler.transform(raw)

    X_test = []
    for i in range(window_size, raw.shape[0]):
        X_test.append(raw[i-window_size:i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

X_test = preprocess_testdat()

predicted_price_ = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_)

# Plot predicted price vs actual closing price 
test['Predictions_lstm'] = predicted_price
Time to calculate the performance metrics and log them to Neptune.

# Evaluate performance
rmse_lstm = calculate_rmse(np.array(test['Close']), np.array(test['Predictions_lstm']))
mape_lstm = calculate_mape(np.array(test['Close']), np.array(test['Predictions_lstm']))

### Neptune new version
npt_exp['RMSE'].log(rmse_lstm)
npt_exp['MAPE (%)'].log(mape_lstm)

### Plot prediction and true trends and log to Neptune         
def plot_stock_trend_lstm(train, test, logNeptune=True):
    fig = plt.figure(figsize = (20,10))
    plt.plot(train['Date'], train['Close'], label = 'Train Closing Price')
    plt.plot(test['Date'], test['Close'], label = 'Test Closing Price')
    plt.plot(test['Date'], test['Predictions_lstm'], label = 'Predicted Closing Price')
    plt.title('LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.legend(loc="upper left")

## Log image to Neptune new version
    if logNeptune:
        npt_exp['Plot of Stock Predictions with LSTM'].upload(neptune.types.File.as_image(fig))

plot_stock_trend_lstm(train, test)

### Stop the run after logging for new version 
npt_exp.stop()