%%
% Load time sereis data from csv
data = readtable("Test_data.csv");
%Change the names VariableNames = {'date,'t1','t2'}

% column_names = data.Properties.VariableNames;

%Convert date to date format
dates = datetime(data.Var1, 'ConvertFrom','datenum');
price_data = table2timetable(data(:,2:end), 'RowTimes', dates);

%% 1. Plotting

%Plot the time series data given
figure;
subplot(1, 2, 1);
plot(data.Var1, data.Var2);
title('T1');
subplot(1, 2, 2)
plot(data.Var1, data.Var3);
title('T2');


%% EDA/Descriptive statistics
%Remove nan values from each and store each times series in separate
%varialbe


%perform exploratory statistical analysis on the data. Data is likely
%distributed non parametrically so quartile tests for outliers will not be
%very effective. Results from EDA will hopefully inform how to deal with
%outliers/noise missing values. High frequecy data. 

%Summary statistic of original data
stats = summary(price_data);
disp(stats.Var2);
disp(stats.Var3);

% Calculate mean, variance, and covariance
m_p = mean(table2array(price_data(:,1:end)), 'omitnan');
v_p = var( table2array(price_data(:,1:end) ), 'omitnan');
% c_p = xcov(table2array(price_data(:,1) ), price_data(:,2), 'biased', 'omitnan');

%Determine the behaviour of the data.

%Stationarity 
% Augmented Dickey-Fuller test (ADF) or Kwiatkowski-Phillips-Schmidt-Shin test (KPSS)
    %Stationarity

% % Test for stationarity using ADF test
% [h1,p1] = adftest(ts1); % First time series
% [h2,p2] = adftest(ts2); % Second time series
% 
% % Display the results of ADF test
% disp('Results of ADF test:');
% disp(['First time series: h = ', num2str(h1), ', p = ', num2str(p1)]);
% disp(['Second time series: h = ', num2str(h2), ', p = ', num2str(p2)]);


%% Classification


%Detrend and demean
    %Trend
    %Corellation between 
    % % 
    % % Decompose time series into trend, seasonality, and residual components
    % data_detrended = detrend(data.Var2);
    % [data_seasonal, seasonal] = seasonaladjust(data_detrended);
    % data_residual = data_detrended - data_seasonal;
    % 
    % % Analyze and model cleaned time series using autocorr, crosscorr, and arima
    % autocorr(data_residual);
    % title('Autocorrelation of Residual Component');
    % 
    % crosscorr(data_detrended, data_seasonal);
    % title('Cross-Correlation between Detrended and Seasonal Components');
    % 
    % model = arima('ARLags', 1, 'MALags', 1);
    % est = estimate(model, data.Var1);
    % residuals = infer(est, data.Var1);
    % plot(residuals);
    % title('Residuals of ARIMA Model');

    % Spearman rank correlation test
    % [rho,pval] = corr(x,y,'Type','Spearman');
    % fprintf('Spearman rank correlation test result: rho = %f, p = %f\n', rho, pval);
    %Covariance


%Hurst exponent
    %Seasonality

%Autocorellation
    %Cyclicity
    % White noise

%Periodogram
    %Seasonality
%smooth data to remove outliers and clean spectrum 
% % Plot the periodogram for each time series
% figure;
% subplot(1,2,1);
% periodogram(t1);
% title('Periodogram of first time series');
% xlabel('Frequency');
% ylabel('Power spectrum density');
% subplot(1,2,2);
% periodogram(t2);
% title('Periodogram of second time series');
% xlabel('Frequency');
% ylabel('Power spectrum density');
%  

% Portmanteau test or Box-Pierce test
    % White noise

% % Test for white noise using Portmanteau test or Box-Pierce test
% [h5,p5] = portmanteautest(ts1); % First time series (Portmanteau)
% [h6,p6] = portmanteautest(ts2); % Second time series (Portmanteau)
% [h7,p7] = boxpiercetest(ts1); % First time series (Box-Pierce)
% [h8,p8] = boxpiercetest(ts2); % Second time series (Box-Pierce)
% 
% % Display the results of Portmanteau and Box-Pierce tests
% disp('Results of Portmanteau and Box-Pierce tests:');
% disp(['First time series: h_portmanteau= ', num2str(h5),', p_portmanteau= ' ,num2str(p5),', h_boxpierce= ' ,num2str(h7),', p_boxpierce= ' ,num2str(p7)]);
% disp(['Second time ser
% 




% % Test for serial correlation using Ljung-Box test
% [h3,p3] = lbqtest(ts1); % First time series
% [h4,p4] = lbqtest(ts2); % Second time series
% 
% % Display the results of Ljung-Box test
% disp('Results of Ljung-Box test:');
% disp(['First time series: h = ', num2str(h3), ', p = ', num2str(p3)]);
% disp(['Second time series: h = ', num2str(h4), ', p = ', num2str(p4)]);
% 



% % Estimate an AR(2) model and find roots of characteristic equation
% [a, e] = arburg(y, 2);
% r = roots(a);
% 
% % Check if any roots are equal to one or close to one
% if any(abs(r - 1) < tol)
%     disp('The time series has a unit root and is likely nonstationary.');
% else
%     disp('The time series is likely a stationary stochastic process.');
% end






%test for normality. 

% a low p-value indicates that the null hypothesis can be rejected and the
% data is not normally distributed.

% Apply Shapiro-Wilk test
[h,p] = swtest(lt1);
disp(['Null hypothesis = ' num2str(h)]);
disp(['Shapiro-Wilk test: p-value = ' num2str(p)]);
[h,p] = swtest(lt2);
disp(['Null hypothesis = ' num2str(h)]);
disp(['Shapiro-Wilk test: p-value = ' num2str(p)]);

% Apply Kolmogorov-Smirnov test
[h,p] = kstest(lt1);
disp(['Null hypothesis = ' num2str(h)]);
disp(['Kolmogorov-Smirnov test: p-value = ' num2str(p)]);
[h,p] = kstest(lt2);
disp(['Null hypothesis = ' num2str(h)]);
disp(['Kolmogorov-Smirnov test: p-value = ' num2str(p)]);

%skewness and kurtosis
skew = skewness(lt1);
skew = skewness(lt2);
kurt = kurtosis(lt1);
kurt = kurtosis(lt2);

% Display the results
disp(['Skewness: ', num2str(skew)])
disp(['Kurtosis: ', num2str(kurt)])




% %Box plot and histogram of log returns
% % Calculate Q1, Q3 and IQR
% Q1 = quantile(t1,0.25);
% Q3 = quantile(t1,0.75);
% IQR = Q3 - Q1;
% 
% % Calculate upper and lower fences
% upper_fence1 = Q3 + 1.5 * IQR;
% lower_fence1 = Q1 - 1.5 * IQR;


% figure;
% subplot(1,2,1);
% boxchart(log_returns_normalized.Fun_Var2, "Orientation","vertical",'JitterOutliers','on');
% % hold on
% % line(xlim,[upper_fence1 upper_fence1],'Color','k','LineStyle','--')
% % line(xlim,[lower_fence1 lower_fence1],'Color','k','LineStyle','--')
% % hold off
% xlabel('Time');
% ylabel('Value');
% title('T1');
% 
% subplot(1,2,2);
% boxchart(log_returns_normalized.Fun_Var3,"Orientation","vertical", 'JitterOutliers','on');
% xlabel('Time');
% ylabel('Value');
% title('T2');
% 
% %Outlier analysis
% idx = isoutlier(log_returns_normalized.Fun_Var2,'quartiles');
% outliers = log_returns_normalized(idx,:);
% disp(['num outliers t1 = ' num2str(size(outliers.Fun_Var2,1)) ]);
% idx = isoutlier(log_returns_normalized.Fun_Var2,'quartiles');
% outliers = log_returns_normalized(idx,:);
% disp(['num outliers t2 = ' num2str(size(outliers.Fun_Var3,1)) ]);
% 
% 
% 




%% Transform Data 
%Since time series stock price data, useful to use logrithmic
%transformation. The distribution of log returns can unlike linear returns
%easily be project to any horizon. Log returns typically have a symmetric
%distribution which makes modelling easier (stock prices are often assumed
%to be log normally distributed - log-returns follow a normal distribution)

% Compute the logarithmic returns of the stock prices
log_returns = price2ret(price_data);

% Plot
figure;
subplot(1,2,1);
plot(log_returns.Time, log_returns.Var2);
xlabel('Time');
ylabel('Value');
title('Log returns T1');
subplot(1,2,2);
plot(log_returns.Time, log_returns.Var3);
xlabel('Time');
ylabel('Value');
title('Log returns T2');


% Normalize the data using Z-Score normalization
log_returns_normalized = varfun(@(x) zscore(x,0,'omitnan'), log_returns);

lt1= log_returns_normalized.Fun_Var2;
lt2 = log_returns_normalized.Fun_Var3;

% Plot logarithmic return ontop of original time series
figure;
subplot(1,2,1);
plot(log_returns_normalized.Time, lt1);
xlabel('Time');
ylabel('Value');
title('Log returns T1');
subplot(1,2,2);
plot(log_returns_normalized.Time, lt2);
xlabel('Time');
ylabel('Value');
title('Log returns T2');

%Summary stats of logarithmic returns 
stats = summary(log_returns_normalized);
disp(stats);


%Verify normality of logarithmic returns. If non-normal then will have to
%use non-parametric tests 
figure;
subplot(1,2,1);
histogram(lt1);
xlabel('Time');
ylabel('Value');
title('T1');
subplot(1,2,2);
histogram(lt2);
xlabel('Time');
ylabel('Value');
title('T2');


 

% close all

%% Cleaning data

t1 = price_data.Var2;
t2 = price_data.Var3;

% t1n = t1(~isnan(t1));
% t2n = t2(~isnan(t2));

% t1s = smoothdata(t1,"rlowess","includenan");
% t2s = smoothdata(t2,"rlowess","includenan");

% Remove outliers, missing values, or noise in the data

% % % Calculate Q1, Q3 and IQR
% Q1 = quantile(t1,0.25);
% Q3 = quantile(t1,0.75);
% IQR = Q3 - Q1;
% 
% % Calculate upper and lower fences
% upper_fence1 = Q3 + 1.5 * IQR;
% lower_fence1 = Q1 - 1.5 * IQR;
% 
% Q1 = quantile(t2,0.25);
% Q3 = quantile(t2,0.75);
% IQR = Q3 - Q1;
% 
% % Calculate upper and lower fences
% upper_fence2 = Q3 + 1.5 * IQR;
% lower_fence2 = Q1 - 1.5 * IQR;
% 
% 
% data.Var2 = rmoutliers(t1,"quartiles","ThresholdFactor",[]);
% data.Var2 = fillmissing(data.Var2,"pchip");
% data.Var2 = smoothdata(data.Var2,"rlowess" );
%remove

% figure;
% plot(data.Var1, data.Var2);
% title('T1');
% figure;
% plot(data.Var1, data.Var3);
% title('T2');



% % Remove outliers
% outlier_idx = isoutlier(data.Value);
% data(outlier_idx, :) = [];
% data.Value(outlier_idx) = interp1(time(~outlier_idx), data.Value(~outlier_idx), time(outlier_idx));
% 
% 
% % Resample the data to a regular time grid with a 10-minute interval
% data_resampled = resample(data.Value, time, 'regular', 10, 'mean');
% 
% % Save the cleaned and resampled data to a new CSV file
% cleaned_data = table(time, data_resampled);
% writetable(cleaned_data, 'cleaned_time_series_data.csv');


% missingValues = 0;
% Data.AirSpeed = standardizeMissing(Data.AirSpeed, missingValues);
% 
% maxDropSpan = 5;
% DataFixed = varfun(@(x)fillSections(x, maxDropSpan), Data);
% DataFixed.Properties.VariableNames = cellstr("Fixed" + Data.Properties.VariableNames);
% Data = [Data DataFixed]; % Append fixed to original



%% Plotting cleaned normalised data
%Calculate and plot normalised log returns

% % Compute the logarithmic returns of the stock prices
% log_returns = price2ret(price_data);
% 
% % Normalize the data using Z-Score normalization
% log_returns_normalized = zscore(log_returns);


%% Model selections

% % Split data into training and testing sets
% c = cvpartition(stock_price,'HoldOut',0.2);
% trainData = stock_price(c.training,:);
% testData = stock_price(c.test,:);
% 
% 
% 
% % Train linear regression model with one lagged variable
% mdl1 = fitlm(trainData,'y ~ lag1(y)');
% 
% % Calculate MSE of linear regression model on testing set
% yhat1 = predict(mdl1, testData);
% mse1 = mean((yhat1 - testData.y).^2);
% 
% % Calculate AIC of linear regression model
% aic1 = aicbic(mdl1.LogLikelihood, numel(mdl1.Coefficients), numel(trainData));
% 
% 
% % Train an ARIMA model on the training set with order (1,1,0)
% mdl2 = arima(1, 1, 0);
% mdl2 = estimate(mdl2, diff(stock_price(c.training,:).Close));






% % Specify an ARIMA(1,1,1) model with a GARCH(1,1) variance model
% varMdl = garch(1,1);
% meanMdl = arima('ARLags',1,'D',1,'MALags',1,'Variance',varMdl);
% 
% % Estimate the model parameters using maximum likelihood
% estMdl = estimate(meanMdl,y);
% 
% % Forecast the conditional mean and variance for 10 steps ahead
% [yF,yMSE,vF,vMSE] = forecast(estMdl,10,'Y0',y);
% 
% % Plot the forecasted mean and variance with 95% confidence intervals
% figure;
% subplot(2,1,1);
% plot(y(end-100:end),'b');
% hold on;
% plot(length(y)+1:length(y)+10,yF,'r','LineWidth',2);
% plot(length(y)+1:length(y)+10,yF+1.96sqrt(yMSE),'k--');
% plot(length(y)+1:length(y)+10,yF-1.96sqrt(yMSE),'k--');
% hold off;
% legend('Historical','Forecast','95% Interval','Location','NorthWest');
% title('Forecasted Conditional Mean');
% subplot(2,1,2);
% plot(vF,'r','LineWidth',2);
% hold on;
% plot(vF+1.96sqrt(vMSE),'k--');
% plot(vF-1.96sqrt(vMSE),'k--');
% hold off;
% legend('Forecast','95% Interval','Location','NorthWest');
% title('Forecasted Conditional Variance');




% % Fit the model to the training set using maximum likelihood
% estMdlTrain = estimate(meanMdl,ytrain);
% 
% % Compute the log-likelihood of the fitted model on the testing set
% [logL] = infer(estMdlTrain,ytest);
% 
% % Perform a 5-fold cross-validation on the whole data set
% cvlogL = cvpartition(logL,n,'KFold',5);
% 
% % Compute the mean log-likelihood for each fold
% meanlogL = zeros(5,1);
% for i = 1:5
% meanlogL(i) = mean(logL(cvlogL.test(i)));
% end
% 
% % Display the mean log-likelihood for each fold and their average
% disp('Mean log-likelihood for each fold:');
% disp(meanlogL);
% disp('Average mean log-likelihood:');
% disp(mean(meanlogL));

%%

% function vFilled = fillSections(v, maxDropSpan)
% % FILLED = FILLSECTIONS(V, MAXDROPSPAN)
%     [vFilled, filledIdx] = fillmissing(v, 'linear');
%     tooLongIdx = bwareaopen(filledIdx, maxDropSpan);    
%     vFilled(tooLongIdx) = NaN;
%     vFilled = fillmissing(vFilled, 'spline');    
% end


% close all

