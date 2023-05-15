# Trad modelling

import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima.model_selection import cross_validate

# Load your time series data into a pandas dataframe
df = pd.read_csv('your_time_series_data.csv')

# Fit an auto arima model to the data
model = auto_arima(df['target_variable'], seasonal=False)

# Define the number of cross-validation folds
n_folds = 5

# Perform cross-validation using the cross_validate function
cv_results = cross_validate(
    model,  # the auto_arima model
    df['target_variable'],  # the target variable
    cv=n_folds,  # number of cross-validation folds
    return_estimator=True,  # return the fitted models
    verbose=2  # print progress messages
)

# Get the order of the best model
best_model_idx = cv_results['test_score'].argmax()
best_model = cv_results['estimator'][best_model_idx]
best_order = best_model.order

# Print the order of the best model
print(f"The order of the best auto_arima model is {best_order}")



model1_cv_scores = model_selection.cross_val_score(
    model1, train, scoring='smape', cv=cv, verbose=2)
print("Model 1 CV scores: {}".format(model1_cv_scores.tolist()))

m1_average_error = np.average(model1_cv_scores)

import pandas as pd
from pmdarima.arima import auto_arima
from pmdarima.model_selection import RollingForecastCV

# Load your time series data into a pandas dataframe
df = pd.read_csv('your_time_series_data.csv')

# Create dummy variables for any categorical variables in your data
dummy_vars = pd.get_dummies(df['categorical_variable'])

# Concatenate the dummy variables with the original data
df = pd.concat([df, dummy_vars], axis=1)

# Define the number of splits for the rolling window cross validation
n_splits = 5

# Define the rolling forecast cross validation object
rfcv = RollingForecastCV(
    horizon=1,  # forecast horizon
    step=1,  # step size
    n_splits=n_splits,  # number of splits
    initial=100  # initial training set size
)

# Iterate over the splits and fit an auto arima model to each split
for train_index, test_index in rfcv.split(df):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]
    model = auto_arima(
        train_data['target_variable'],  # target variable
        exogenous=train_data.drop('target_variable', axis=1),  # exogenous variables
        seasonal=False
    )
    model.fit(train_data['target_variable'], exogenous=train_data.drop('target_variable', axis=1))
    # Evaluate the model on the test data
    predictions = model.predict(n_periods=len(test_data), exogenous=test_data.drop('target_variable', axis=1))
    # Do something with the predictions




model_fit = model.fit(trend='nc', disp=0)
yhat = model_fit.forecast()[0]
yhat = inverse_difference(history, yhat, months_in_year)
predictions.append(yhat)
history.append(test[t])
# calculate out of sample error
rmse = sqrt(mean_squared_error(test, predictions))
return rmse


# errors
residuals = [test[i]-predictions[i] for i in range(len(test))]
residuals = DataFrame(residuals)
print(residuals.describe())
# plot
pyplot.figure()
pyplot.subplot(211)
residuals.hist(ax=pyplot.gca())
pyplot.subplot(212)
residuals.plot(kind='kde', ax=pyplot.gca())
pyplot.show()

rmse = evaluate_arima_model(dataset, order)
if rmse < best_score:
best_score, best_cfg = rmse, order


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
predictions = list()
# split dataset
train, test = train_test_split(data, n_test)
# seed history with training dataset
history = [x for x in train]
# step over each time step in the test set
for i in range(len(test)):
# fit model and make forecast for history
yhat = simple_forecast(history, cfg)
# store forecast in list of predictions
predictions.append(yhat)
# add actual observation to history for the next loop
history.append(test[i])
# estimate prediction error
error = measure_rmse(test, predictions)
return error


## ARIMA
model = pm.auto_arima(train, seasonal=True, m=52)
preds = model.predict(test.shape[0])
# You can show them with the plot that is created here:

x = np.arange(y.shape[0])
plt.plot(co2_data.co2.values[:2200], train)
plt.plot(co2_data.co2.values[2200:], preds)
plt.show()



# forecast
forecast, stderr, conf = model_fit.forecast()
# summarize forecast and confidence intervals
print('Expected: %.3f' % test[0])
print('Forecast: %.3f' % forecast)
print('Standard Error: %.3f' % stderr)
print('95%% Confidence Interval: %.3f to %.3f' % (conf[0][0], conf[0][1]))

model = ARIMA(train, order=(5,1,1))
model_fit = model.fit(disp=False)
# plot some history and the forecast with confidence intervals
model_fit.plot_predict(len(train)-10, len(train)+1)
pyplot.legend(loc='upper left')
pyplot.show()



modelARIMA_tuned=ARIMA(endog=Y_train,exog=X_train_ARIMA,order=[2,0,1])
model_fit_tuned = modelARIMA_tuned.fit()
# estimate accuracy on validation set
predicted_tuned = model_fit.predict(start = tr_len -1 ,\
  end = to_len -1, exog = X_test_ARIMA)[1:]
print(mean_squared_error(Y_test,predicted_tuned))



# plotting the actual data versus predicted data
predicted_tuned.index = Y_test.index
pyplot.plot(np.exp(Y_test).cumprod(), 'r', label='actual',)

# plotting t, a separately
pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b--', label='predicted')
pyplot.legend()
pyplot.rcParams["figure.figsize"] = (8,5)
pyplot.show()




# We can compute predictions the same way we would on a normal ARIMA object:
preds, conf_int = pipe.predict(n_periods=10, return_conf_int=True)
print("\nForecasts:")
print(preds)

# Let's take a look at the actual vs. the predicted values:
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.tight_layout()

# Visualize goodness of fit
in_sample_preds, in_sample_confint = \
    pipe.predict_in_sample(X=None, return_conf_int=True)

n_train = train.shape[0]

x0 = np.arange(n_train)
axes[0].plot(x0, train, alpha=0.75)
axes[0].scatter(x0, in_sample_preds, alpha=0.4, marker='x')
axes[0].fill_between(x0, in_sample_confint[:, 0], in_sample_confint[:, 1],
                     alpha=0.1, color='b')
axes[0].set_title('Actual train samples vs. in-sample predictions')
axes[0].set_xlim((0, x0.shape[0]))

# Visualize actual + predicted
x1 = np.arange(n_train + preds.shape[0])
axes[1].plot(x1[:n_train], train, alpha=0.75)
# axes[1].scatter(x[n_train:], preds, alpha=0.4, marker='o')
axes[1].scatter(x1[n_train:], test[:preds.shape[0]], alpha=0.4, marker='x')
axes[1].fill_between(x1[n_train:], conf_int[:, 0], conf_int[:, 1],
                     alpha=0.1, color='b')
axes[1].set_title('Actual test samples vs. forecasts')
axes[1].set_xlim((0, data.shape[0]))

# We can also call `update` directly on the pipeline object, which will update
# the intermittent transformers, where necessary:
newly_observed, still_test = test[:15], test[15:]
pipe.update(newly_observed, maxiter=10)

# Calling predict will now predict from newly observed values
new_preds = pipe.predict(still_test.shape[0])
print(new_preds)

x2 = np.arange(data.shape[0])
n_trained_on = n_train + newly_observed.shape[0]

axes[2].plot(x2[:n_train], train, alpha=0.75)
axes[2].plot(x2[n_train: n_trained_on], newly_observed, alpha=0.75, c='orange')
# axes[2].scatter(x2[n_trained_on:], new_preds, alpha=0.4, marker='o')
axes[2].scatter(x2[n_trained_on:], still_test, alpha=0.4, marker='x')
axes[2].set_title('Actual test samples vs. forecasts')
axes[2].set_xlim((0, data.shape[0]))

plt.show()


















### plot reisduals of arima model
#acf pcf qq and probability plot
#fit garch with ideal arima paramters
# Financial time-series have tails that are heavier than implied by a GARCH process with Gaussian {ϵ(t)}. To handle such data, one can assume that, instead of being Gaussian white noise, {ϵ(t)} is i.i.d. white noise process with a heavy-tailed distribution.
# In fact, GARCH processes exhibit heavy tails even if {ϵ(t)} is Gaussian. Therefore, when we use GARCH models, we can model both the conditional heteroskedasticity and the heavy-tailed distributions of financial markets data.

#GARCh
# To model the conditional change in variance over time.
# Changes in the time-dependent variance.
       for s, t in enumerate(range(trainsize, T-1)):
            train_set = data.iloc[s: t]
            test_set = data.iloc[t+1]  # 1-step ahead forecast
            model = arch_model(y=train_set, p=p, q=q).fit(disp='off')
            forecast = model.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            result.append([(test_set-mu)**2, var])
        df = pd.DataFrame(result, columns=['y_true', 'y_pred'])
        results[(p, q)] = np.sqrt(mean_squared_error(df.y_true, df.y_pred))
        
        best_p, best_q = 2, 2,
am = ConstantMean(nasdaq_returns.clip(lower=nasdaq_returns.quantile(.05),
                                      upper=nasdaq_returns.quantile(.95)))
am.volatility = GARCH(best_p, 0, best_q)
am.distribution = Normal()
best_model = am.fit(update_freq=5)
print(best_model.summary())
Check Residuals
fig = best_model.plot(annualize='D')
fig.set_size_inches(12, 8)
fig.tight_layout();
plot_correlogram(best_model.resid.dropna(),
                 lags=250,
                 title='GARCH Residuals')



model = arch_model(train, mean='Zero', vol='GARCH', p=15, q=15)
# fit model
model_fit = model.fit()
# forecast the test set
yhat = model_fit.forecast(horizon=n_test)
# plot the actual variance
var = [i*0.01 for i in range(0,100)]
pyplot.plot(var[-n_test:])
# plot forecast variance
pyplot.plot(yhat.variance.values[-1, :])
pyplot.show()









#df_transformed = df_transformed.apply(minmax_scale)
model = VARMAX(df_transformed.loc[:'2017'], order=(1,1), trend='c').fit(maxiter=1000)

print(model.summary())
model.plot_diagnostics(variable=0, figsize=(14,8), lags=24)
plt.gcf().suptitle('Industrial Production - Diagnostics', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=.93);

#predictions
n =len(df_transformed)
start = n-24

preds = model.predict(start=start+1, end=n)
preds.index = df_transformed.index[start:]

fig, axes = plt.subplots(nrows=2, figsize=(14, 8), sharex=True)

df_transformed.ip.loc['2010':].plot(ax=axes[0], label='actual', title='Industrial Production')
preds.ip.plot(label='predicted', ax=axes[0])
trans = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
axes[0].legend()
axes[0].fill_between(x=df_transformed.index[start+1:], y1=0, y2=1, transform=trans, color='grey', alpha=.5)

trans = mtransforms.blended_transform_factory(axes[0].transData, axes[1].transAxes)
df_transformed.sentiment.loc['2010':].plot(ax=axes[1], label='actual', title='Sentiment')
preds.sentiment.plot(label='predicted', ax=axes[1])
axes[1].fill_between(x=df_transformed.index[start+1:], y1=0, y2=1, transform=trans, color='grey', alpha=.5)
axes[1].set_xlabel('')
sns.despine()
fig.tight_layout();





# Compare predictions to actual load
eval_df = pd.DataFrame(predictions, 
columns=['t+'+str(t) for t in range(1, HORIZON+1)])
eval_df['timestamp'] = test.index[0:len(test.index)-HORIZON+1]
eval_df = pd.melt(eval_df, id_vars='timestamp', 
value_name='prediction', var_name='h')
eval_df['actual'] = np.array(np.transpose(test_ts)).ravel()
eval_df[['prediction', 'actual']] = scaler.inverse_transform(eval_
df[['prediction', 'actual']])