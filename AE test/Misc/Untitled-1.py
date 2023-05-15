

import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
import numpy as np

models = {
    'model_1': model_1_predict_function,
    'model_2': model_2_predict_function,
    # ... add other models here
}

datasets = [data_set_1, data_set_2]



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / ((y_true + y_pred)/2))) * 100

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'msle': msle,
        'smape': smape
    }


def residual_analysis(y_true, y_pred):
    residuals = y_true - y_pred
    # Perform any required residual analysis here
    # ...
    return residual_analysis_results

def model_selection(model_evaluations):
    best_model = None
    best_score = float('inf')
    for model_eval in model_evaluations:
        if model_eval['evaluation_score'] < best_score:
            best_score = model_eval['evaluation_score']
            best_model = model_eval['model_name']
    return best_model

def cross_val(X, y, predict_func, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
            # Make predictions on the test data
            y_pred = predict_func(X_train, y_train, X_test)

            # Evaluate the model
            mse = evaluate(y_test, y_pred)
            scores.append(mse)

    return np.array(scores)




    model_evaluations = []

    # Iterate through all models and dataset combinations for training/testing, evaluation, cross-validation, and residual analysis
    for model_name, predict_func in models.items():
        for train_data, test_data in itertools.product(datasets, repeat=2):
            # Split the data into features and labels
            X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
            X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
            # Train and test the model
            y_pred = predict_func(X_train, y_train, X_test)

            # Evaluate the predictions
            evaluation_score = evaluate(y_test, y_pred)

            # Perform cross-validation
            cv_scores = cross_val(X_train, y_train, predict_func)
            mean_cv_score = cv_scores.mean()
            std_dev = cv_scores.std()

            # Perform residual analysis
            residual_analysis_results = residual_analysis(y_test, y_pred)

            # Output the evaluation, CV, and residual analysis results
            print(f"Model: {model_name}, Train data: {train_data}, Test data: {test_data}")
            print(f'Model: {model_name}, Dataset Combination: {dataset_combination}, MAE: {evaluation_score["mae"]}, MSE: {evaluation_score["mse"]}, RMSE: {evaluation_score["rmse"]}, MAPE: {evaluation_score["mape"]}, R2: {evaluation_score["r2"]}, MSLE: {evaluation_score["msle"]}, sMAPE: {evaluation_score["smape"]}')
            print(f"Mean cross-validation score: {mean_cv_score}, Standard deviation: {std_dev}")
            print(f"Residual analysis results: {residual_analysis_results}")
            print("\n")


    # Append the evaluation results to the model_evaluations list
    model_evaluations.append({
        'model_name': model_name,
        'train_data': train_data,
        'test_data': test_data,
        'evaluation_score': evaluation_score,
        'mean_cv_score': mean_cv_score,
        'std_dev': std_dev,
        'residual_analysis_results': residual_analysis_results
    })








# Here's an outline of the process:

# 1. Define a dictionary of models and their corresponding predict functions.
# 2. Define a list of datasets to test different hypotheses.
# 3. Define functions for evaluation, kfold cross validation(with mean and std) residual analysis, and model selection.
# 4. Iterate through all models and dataset combinations for training/testing, evaluation, cross-validation, and residual analysis.
# 5. Output the evaluation, cross-validation, and residual analysis results for each model and dataset combination.
# 6. Perform model selection based on the evaluation results.
# 7. Print the best model.

# This code will perform the evaluation, cross-validation, residual analysis, and model selection for each model and dataset combination. The best model is then selected based on the evaluation results.



# Determine the best model based on the cross-validation results
def select_best_model(model_evaluations, criterion='rmse'):
    best_model = None
    best_evaluation = float('inf')
    for evaluation in model_evaluations:
        if evaluation[criterion] < best_evaluation:
            best_evaluation = evaluation[criterion]
            best_model = evaluation['model_name']
    return best_model

# Initialize a dictionary to store the aggregated evaluation results
aggregated_evaluations = {}

# Iterate through all models and dataset combinations for training/testing and evaluation
model_evaluations = []
for model_name, predict_funct in models.items():
    for dataset_combination in dataset_combinations:
        # Split the data into features and labels
        X, y = dataset_combination[0], dataset_combination[1]

        # Initialize the evaluation metrics' sum for the current model and dataset combination
        evaluation_sum = defaultdict(float)
        n_folds = 0

        # Iterate through the rolling forecast origin split
        for X_train, y_train, X_test, y_test in rolling_forecast_origin_split(X, y, window_size=30):
            # Train and test the model
            y_pred = predict_funct(X_train, y_train, X_test)
            # Evaluate the model
            evaluation_result = evaluate(y_test, y_pred)
            
            # Update the evaluation metrics' sum
            for metric, value in evaluation_result.items():
                evaluation_sum[metric] += value
            n_folds += 1

        # Calculate the evaluation metrics' mean
        evaluation_mean = {metric: value / n_folds for metric, value in evaluation_sum.items()}
            
        # Print the mean evaluation metrics for the current model and dataset combination
        print(f'Model: {model_name}, Dataset Combination: {dataset_combination}, Mean Evaluation Metrics: {evaluation_mean}')
        # Append the mean evaluation results to the model_evaluations list
        evaluation_mean.update({'model_name': model_name, 'dataset_combination': dataset_combination})
        model_evaluations.append(evaluation_mean)
        
    # Select the best model based on the lowest mean RMSE (or another evaluation metric)
    best_model_name = select_best_model(model_evaluations, criterion='rmse')
    print(f'The best model based on cross-validation is {best_model_name}.')


    
  



Plot histogram of missing values streaks
def plot_streak_lengths(ts, title): 
    isnull = ts.isnull()
    groups = isnull.ne(isnull.shift()).cumsum()
    streak_lengths = isnull.groupby(groups).sum()
    plt.hist(streak_lengths[(streak_lengths > 0) & (streak_lengths < 100)].values, bins=100)
    plt.xlabel('Streak length')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def plot_streak_histogram_1(ts, title):
    isnull = ts.isnull()
    groups = isnull.ne(isnull.shift()).cumsum()
    streak_lengths = isnull.groupby(groups).sum()

    bins = pd.cut(streak_lengths, bins=[0, 1, 2, 3, 4, 5, 10, 100, 1000, streak_lengths.max()])
    plt.figure(figsize=(8, 7))
    # plot histogram of binned data
    ax = bins.value_counts().sort_index().plot(kind='bar', edgecolor='black')
    plt.title(title)
    # add values to bars
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    # show the plot
    plt.show()

def plot_streak_histogram_1(ts, title):
    isnull = ts.isnull()
    groups = isnull.ne(isnull.shift()).cumsum()
    streak_lengths = isnull.groupby(groups).sum()
    print(streak_lengths)
    bins = pd.cut(streak_lengths, bins=[0, 1, 2, 3, 4, 5, 10, 100, streak_lengths.max()+1])
    plt.figure(figsize=(8, 7))
    # plot histogram of binned data
    ax = bins.value_counts(sort=False).plot(kind='bar', edgecolor='black')
    ax.set_title(title)  # changed to ax.set_title()

    # add values to bars
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    # show the plot
    plt.show()





### Old imputaiton methods
# def impute_missing_values(series, window=240):
#     if series.isnull().any():
#         rolling_mean = series.rolling(window=window, min_periods=1).mean()
#         imputed_series = series.fillna(rolling_mean)
#         imputed_series.fillna(method='bfill', inplace=True)  # Backfill since missing starting data
#     else:
#         imputed_series = series
#     return imputed_series

# # Impute missing values in ts1 and ts2 using their respective maximum missing ranges
# imputed_ts1 = impute_missing_values(ts1, max_ts1)
# imputed_ts2 = impute_missing_values(ts2, max_ts2)

# # Print the number of missing values in the imputed series
# print(f"Missing values in imputed ts1: {imputed_ts1.isna().sum()}")
# print(f"Missing values in imputed ts2: {imputed_ts2.isna().sum()}")

# # Calculate the maximum missing range in the imputed series
# max_imputed_ts1 = max_missing_range(imputed_ts1)
# max_imputed_ts2 = max_missing_range(imputed_ts2)

# # Print the maximum missing range in the imputed series
# print(f"Max missing range in imputed ts1: {max_imputed_ts1}")
# print(f"Max missing range in imputed ts2: {max_imputed_ts2}")
# #PLot original data
# # plot_time_series(ts1, ts2)

# # Plot the imputed time series
# plot_time_series(imputed_ts1, imputed_ts2)



# Change points and structural breaks analysis:
# - Testing for change points or structural breaks in the time series data using methods such as the Chow test, CUSUM test, or Bai-Perron test
# - Visualizing the identified change points or breaks on the time series plot to better understand the structural shifts in the data
# - Investigating the potential causes of the structural breaks, such as major economic events, changes in market regulations, or other external factors





# Hurst exponent
# from numba import jit

# @jit(nopython=True)
def hurst_exponent_fast(ts):
    lags = np.arange(195, 975) # day to weekly
    tau = np.array([np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags])
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

print('Hurst Exponent (Fast):', hurst_exponent_fast(rs_1))
print('Hurst Exponent (Fast):', hurst_exponent_fast(rs_2))


from statsmodels.stats.diagnostic import acorr_ljungbox

# def portmanteau_test(series, lags=10):
#     result = acorr_ljungbox(series, lags=lags)
#     return pd.DataFrame({'Lag': range(1, lags + 1), 'Test Statistic': result[0], 'p-value': result[1]})

def white_noise_test(series, lags=10):
    result = portmanteau_test(series, lags)
    white_noise = result['p-value'] > 0.05
    return white_noise.all()    
    

def white_noise_test(series):
    lbvalue, pvalue = sm.stats.diagnostic.acorr_ljungbox(series, lags=[1], boxpierce=True)
    result = {'lbvalue': lbvalue[0], 'pvalue': pvalue[0]}
    return result


white_noise_pvalue = white_noise_test(imputed_ts1)['pvalue']
white_noise_pvalue = white_noise_test(imputed_ts2)['pvalue']

# Portmanteau test
result = portmanteau_test(rt['ts1'])
print(result)
result = portmanteau_test(rt['ts2'])
print(result)


# White noise test
is_white_noise = white_noise_test(rt)
print(is_white_noise)


def cap_outliers(series, upper_threshold):
    # Calculate the upper threshold based on the minimum value of the outlier detection output
    # upper_threshold = detect_outliers_func(series).min()

    # Cap the values in the data based on the upper threshold
    capped_data = series.copy()
    capped_data[np.abs(capped_data) > upper_threshold] = upper_threshold

    # Return the capped data
    return capped_data



# def cap_outliers(series, detect_outliers_func):
#     # Calculate the upper threshold based on the minimum value of the outlier detection output
#     upper_threshold = detect_outliers_func(series).min()

#     # Cap the values in the data based on the upper threshold
#     capped_data = series.copy()
#     capped_data[capped_data > upper_threshold] = upper_threshold

#     # Return the capped data
#     return capped_data


# # Define a function to filter outliers in both series
# def filter_common_outliers(series1, series2, outlier_detection_method):
#     # Identify the outliers in both series using the outlier detection method
#     outliers_series1 = outlier_detection_method(series1)
#     outliers_series2 = outlier_detection_method(series2)

#     # Find common outliers
#     common_outliers = np.intersect1d(outliers_series1, outliers_series2)

#     # Filter non-common outliers from both series
#     filtered_series1 = np.setdiff1d(series1, np.setdiff1d(outliers_series1, common_outliers))
#     filtered_series2 = np.setdiff1d(series2, np.setdiff1d(outliers_series2, common_outliers))

#     # Combine the filtered series with the common outliers
#     combined_series1 = np.concatenate((filtered_series1, common_outliers))
#     combined_series2 = np.concatenate((filtered_series2, common_outliers))

#     return combined_series1, combined_series2

# # Set the outlier detection method to be extreme_value_analysis
# detect_outliers = extreme_value_analysis

# # Filter outliers in both series using cap_outliers method and detect_outliers as argument
# capped_ts1 = cap_outliers(rt_1, detect_outliers)
# capped_ts2 = cap_outliers(rt_2, detect_outliers)

# # Filter common outliers from both series using filter_common_outliers function
# combined_ts1, combined_ts2 = filter_common_outliers(capped_ts1, capped_ts2, outlier_detection_method=detect_outliers)

# # Combine the filtered series
# c_rt = pd.concat([combined_ts1, combined_ts2], axis=1)


# Residuals analysis: evaluate the data analysis methods applied above
# 1. Plot residuals between data anad output
# 2. Serial correlation analysis: autocorrelation function (ACF) and partial autocorrelation function (PACF) to check for correlation between residuals at different lags. If there is significant correlation, it indicates that the model may not be capturing all relevant patterns in the data.
# 3. White noise analysis is another method of residual analysis that involves testing whether the residuals are random and uncorrelated. This can be done using the Ljung-Box test or Breusch-Godfrey test. If the test indicates that there is significant correlation, it suggests that the model may not be appropriate for the data.


#    - After decomposing the time series into trend, seasonal, and residual components, perform residual analysis on the residual component to check for any patterns or trends that were not captured by the model.
#    Plot the residuals to visualize their distribution and check for any outliers or anomalies.
#    Seasonality analysis:

#    - After performing autocorrelation analysis, seasonal decomposition, or spectral analysis, perform residual analysis on the seasonally adjusted data to check for any patterns or trends that were not captured by the seasonal component of the model.
#    Plot the residuals to visualize their distribution and check for any outliers or anomalies.
#    Cyclicity analysis:

#    - After performing autocorrelation analysis or spectral analysis, perform residual analysis on the data to check for any patterns or trends that were not captured by the model.
#    Plot the residuals to visualize their distribution and check for any outliers or anomalies.

