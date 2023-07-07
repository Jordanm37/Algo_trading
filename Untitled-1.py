# %%
from modelling_functions import *

# %%
models_and_param_grids = [
    {
        'model': DecisionTreeRegressor(random_state=42),
        'param_grid': {
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto'],
            'min_samples_leaf': [1, 3, 5, 10]
        },
        'model_name': 'CART',
    },
    {
        'model': KNeighborsRegressor(),
        'param_grid': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],
            'leaf_size': [10, 30, 50],
        },
        'model_name': 'KNN',
    },
    {
        'model': lgb.LGBMRegressor(max_depth=-1, random_state=42),
        'param_grid': {
            'lgbmregressor__n_estimators': [100, 200],
            'lgbmregressor__learning_rate': [0.01],
            'lgbmregressor__max_depth': [5, 10, 20],
            'lgbmregressor__num_leaves': [35, 50],
        },
        'model_name': 'GBR',
    },
    {
        'model': xgb.XGBRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 3, 5],
        },
        'model_name': 'XGB',
    },
    {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto'],
        },
        'model_name': 'RF',
    },
    # {
    #     'model': MLPRegressor(random_state=42),
    #     'param_grid': {
    #         'hidden_layer_sizes': [(50,), (100,), (50, 50),(100, 100), (50, 50, 50), (100, 100, 100), (50, 50, 50, 50), (100, 100, 100, 100)],
    #         'activation': ['tanh', 'relu', 'logistic'],
    #         'solver': ['sgd'],
    #         'alpha': [0.00005, 0.0005, 0.005],
    #         'early_stopping': [True],
    #         'max_iter': [600],
    #         'shuffle': [False],
    #     },
    #     'model_name': 'MLP',
    # },
    #     {
    #     'model': GaussianProcessRegressor(random_state=42),
    #     'param_grid': {
    #         'kernel': [RBF(), DotProduct()+ WhiteKernel()],
    #         'alpha': [1e-10, 1e-5, 1e-2, 1],
    #         'n_restarts_optimizer': [0, 1, 3],
    #     },
    #     'model_name': 'GPR',
    # },
]

file_name_raw = "Test_data.csv"
file_name_clean = "imputed_df.csv"

# load data, transform and split
ts_raw = load_data_file(file_name_raw)
ts_imputed = load_data_file(file_name_clean)

# %% [markdown]
# Calculate naive benchmark using MC simulation

# %%
frequency = 'D'
n_trials = 100
#default process is false for log returns
ts_raw_transformed = transform_data(ts_raw, frequency, do_log_returns=False)
ts_imputed_transformed = transform_data(ts_imputed, frequency, do_log_returns=False)

# %% [markdown]
# Benchmark prediction using geometric brownian motion. 
# S[t] = S[t-1] * (1 + ε) 
# S[t] is the stock price at time t
# S[t-1] is the stock price at time t-1
# ε is a normally distributed random variable with mean 'mu' and standard deviation 'std'      
# 

# %%
run_predictions_and_plot(ts_raw_transformed, n_trials)
run_predictions_and_plot(ts_imputed_transformed, n_trials)


# %% [markdown]
# Model fitting, training and testing

# %%
split_date = '2011-12-31'
frequency = 'D'
LOG_RETURNS = True


test_size = 14
n_splits = 30
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
lags = 3

ts_raw_transformed = transform_data(ts_raw, frequency, do_log_returns=LOG_RETURNS)
ts_imputed_transformed = transform_data(ts_imputed, frequency, do_log_returns=LOG_RETURNS)
train_raw, test_raw = split_data(ts_raw_transformed, split_date)
train_imp, test_imp = split_data(ts_imputed_transformed, split_date)

# %% [markdown]
# Hypotheses models testing for four hypotheses with four cases of features:
# - missing vlaues removed without outliers marked
# - missing values removed with outliers marked
# - missing values imputed without outliers marked
# - missing values imputed with outliers marked

# %%
h1_results = test_hypothesis('H1', 'ts1', 'ts1', models_and_param_grids, tscv, lags, train_raw, test_raw, train_imp, n_splits, test_size)
h2_results = test_hypothesis('H2', 'ts2', 'ts2', models_and_param_grids, tscv, lags, train_raw, test_raw, train_imp, n_splits, test_size)
h3_results = test_hypothesis('H3', 'ts1', 'ts2', models_and_param_grids, tscv, lags, train_raw, test_raw, train_imp, n_splits, test_size)
h4_results = test_hypothesis('H4', 'ts2', 'ts1', models_and_param_grids, tscv, lags, train_raw, test_raw, train_imp, n_splits, test_size)

plot_all_experiment_results_comparison(h1_results, 'H1')
plot_all_experiment_results_comparison(h2_results, 'H2')
plot_all_experiment_results_comparison(h3_results, 'H3')
plot_all_experiment_results_comparison(h4_results, 'H4')



