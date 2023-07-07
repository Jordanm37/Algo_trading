
from eda_fucntions import *


# Load imputed data and calculate returns
ts1_i, ts2_i, rt1_i, rt2_i = load_data_and_calculate_returns(
    'interpolate_clean_df.csv', imputed=True)
# Load non-imputed data and calculate returns
ts1_o, ts2_o, rt_1_o, rt_2_o = load_data_and_calculate_returns(
    "Test_data.csv", imputed=False)


# plot normalised time series
plot_normalised_price(ts1_o, ts2_o)

# non-imputed data
plot_series_and_returns(ts1_o, 'ts1_raw')
plot_series_and_returns(ts2_o, 'ts2_raw')


# analyse time series


# non-imputed data
ts1 = ts1_o
ts2 = ts2_o
rt1 = rt_1_o
rt2 = rt_2_o


#seasonality analysis
print("Seasonality analysis for ts1:")
analyse_seasonality(ts1, 'ts1', freq_list=['H', 'D', 'W'])

print("Seasonality analysis for ts2:")
analyse_seasonality(ts2, 'ts2', freq_list=['H', 'D', 'W'])



#spectral analysis
print("Spectral analysis for rt1:")
spectral_analysis(rt1, 'ts1', freq_list=['H', 'D', 'W'])

print("Spectral analysis for rt2:")
spectral_analysis(rt2, 'ts2', freq_list=['H', 'D', 'W'])


# Analyse distribution of log returns
# - univariate statistics
# - rolling statistics
# - box plot
# - normality
# - stationarity


#distribution analysis
rt = pd.concat([rt1, rt2], axis=1)
print("Distribution analysis for log returns:")
analyse_distribution(rt)


# 


#volatility clustering analysis
print("Volatility clustering analysis for rt1:")
assess_and_plot_volatility_clustering(rt1)

print("Volatility clustering analysis for rt2:")
assess_and_plot_volatility_clustering(rt2)


# 


#outlier analysis
print("Outlier analysis for rt1 and rt2:")
rt1_d = rt1.resample('H').mean().dropna()
rt2_d = rt2.resample('H').mean().dropna()
analyse_outliers(rt1_d, rt2_d, 'Hourly')
rt1_d = rt1.resample('D').mean().dropna()
rt2_d = rt2.resample('D').mean().dropna()
analyse_outliers(rt1_d, rt2_d, 'Daily')


# imputed data
plot_series_and_returns(ts_1, 'ts1_imputed')
plot_series_and_returns(ts_2, 'ts2_imputed')


# analyse time series


# imputed data
ts1 = ts1_i
ts2 = ts2_i
rt1 = rt1_i
rt2 = rt2_i


#seasonality analysis
print("Seasonality analysis for ts1:")
analyse_seasonality(ts1, 'ts1', freq_list=['H', 'D', 'W'])

print("Seasonality analysis for ts2:")
analyse_seasonality(ts2, 'ts2', freq_list=['H', 'D', 'W'])



#spectral analysis
print("Spectral analysis for rt1:")
spectral_analysis(rt1, 'ts1', freq_list=['H', 'D', 'W'])

print("Spectral analysis for rt2:")
spectral_analysis(rt2, 'ts2', freq_list=['H', 'D', 'W'])


#distribution analysis
rt = pd.concat([rt1, rt2], axis=1)
print("Distribution analysis for log returns:")
analyse_distribution(rt)


#volatility clustering analysis
print("Volatility clustering analysis for rt1:")
assess_and_plot_volatility_clustering(rt1)

print("Volatility clustering analysis for rt2:")
assess_and_plot_volatility_clustering(rt2)


#outlier analysis
print("Outlier analysis for rt1 and rt2:")
rt1_d = rt1.resample('H').mean().dropna()
rt2_d = rt2.resample('H').mean().dropna()
analyse_outliers(rt1_d, rt2_d, 'Hourly')
rt1_d = rt1.resample('D').mean().dropna()
rt2_d = rt2.resample('D').mean().dropna()
analyse_outliers(rt1_d, rt2_d, 'Daily')


# imputed data
plot_series_and_returns(ts_1, 'ts1_imputed')
plot_series_and_returns(ts_2, 'ts2_imputed')





