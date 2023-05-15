
%% load and cleanup data
% load time series data from .csv file
data = readtable("Test_data.csv");
% change name of variables (column names) to something meaningfull
data.Properties.VariableNames = {'date','name1','name2'};
% changing date format
data.date = datetime(data.date, 'ConvertFrom','datenum');
% cleanup data by removing nan entries. since data has nan in column 2 and
% 3 at different dates, it might be better to remove nan entries for each
% column seperately. Otherwise, you can remove nan for entire data set

% creating two new tables with the date + one other column
T1 = data(:,1:2);
T2 = data(:,[1,3]);

% removing nan entries using rmmissing function
T1 = rmmissing(T1);
T2 = rmmissing(T2);

% can change to timetable if you want, but not necssary for analysis 
% T1 = table2timetable(T1);
% T2 = table2timetable(T2);

%% plotting results in two different ways
% using stackedplot of time tables
figure; 
stackedplot(table2timetable(T1));
figure;
stackedplot(table2timetable(T2));
% using columns of table. This method gives a lot more flexibility with
% formatting and figure minipulation than a stacked plot
figure;
subplot(1, 2, 1);
plot(T1.date, T1.name1);
title('T1');
subplot(1, 2, 2)
plot(T2.date, T2.name2);
title('T2');

%% calculating return
return1 = price2ret(T1.name1);
return2 = price2ret(T2.name2);
% plotting
% plotPriceReturn(T1, return1);
% plotPriceReturn(T2, return2);
figure;
plot(T1.date,T1.name1)
ylabel("Price")
yyaxis right
plot(T1.date(1:end-1),return1)
ylabel("Return")
% plotting
figure;
plot(T2.date,T2.name2)
ylabel("Price")
yyaxis right
plot(T2.date(1:end-1),return2)
ylabel("Return")


function plotPriceReturn(T1, return1)
    figure;
    plot(T1.date,T1.name1)
    ylabel("Price")
    yyaxis right
    plot(T1.date(1:end-1),return1)
    ylabel("Return")
end






