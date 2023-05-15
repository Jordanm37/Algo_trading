Data preparation
- Mising value analysis
  - MCAR MAR mNAR
  - Distribution of missing values
- Imputation methods


Exploratory Data analysis
- Decomposition and seasonality analysis
  - STL
  - Periodogram
  - Autocorellation/PCF
  - QQ/Prob plot
- Cyclicity analysis
    - Moving averages
- Statistics of returns
  - Anlalyse distribution
    - Summary statistics
    - Skewness/Kurtosis
    - Assess normality
    - Outlier analysis
  - Test stationarity
    - ADF/KPP test
    - Rolling mean
  - Analyse volatility clustering
    - volatility plot
    - acf/pcf
    - QQ/Prob plot
    - LjungBox TEst

Preprocessing (optional)
  - Compare imputation methods for least impact on statistics


Modelling Chapter
- Data preparation
  - Apply imputation and cleaning methods
  - Transform to returns
  - Resample
  - Split training and holdout
- Hypotheses to test
  - univariate prediction: H1, H2
  - mulitivariate prediction: H3, H4

- Model selection
  - Evaluation metrics
  - Cross validation and splitting method
  - Method for evaluating and comparison
    - All models
        a. Feature engineering 
        b. Training
        c. Testing
        d. Evaluation
        e. Cross validation
        f. Residual and diagnostic
    g. Optimization of best model
    h. Residual and diagnostic** Save plot output from each method to file and folder according to method
  - Testing
    - Univariate: H1, H2
      - Bechmarking
      - Traditional
        - ARIMA
        - GARCH (depending on correlation of residuals fo ARIMA)
      - ML
        - RF
        - SVM
        - LightGBM
        - LSTM
    - Multivariate: H3, H4 
      - Benchmarking
      - Traditional 
        - VAR
      - ML
        - RF
        - SVM
        - LightGBM
        - LSTM   
  - Model selection
    - Select best models  
    - Test on validation set (4 tests)
- Discussion





Rolling window cv

Grid search optimsation with CV



forecasting long term behaviour in test set 
so use entire training set to predict next values

to improve may use single step forecast to better capture volatility