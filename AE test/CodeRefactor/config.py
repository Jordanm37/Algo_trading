# config.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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



split_date = '2011-12-31'
test_size = 14
n_splits = 30
lags = 3
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)