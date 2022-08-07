%Yi Li June 2022
% Import Data
Data_path = 'C:\...\Copula\Data\';  
Data = xlsread([Data_path,'changes.xlsx'],'Changes', 'A2:E648');
Dates = Data(:,1);  
Returns = Data(:,2:end);
T = length(Returns);  % 647
N = size(Returns, 2);  % 4

% Statistic Table
Statist = nan(4,N);
Statist(1,:) = mean(Returns);
Statist(2,:) = std(Returns);
Statist(3,:) = skewness(Returns);
Statist(4,:) = kurtosis(Returns);

% Correlation Table
Corr = zeros(N,N);
Corr = corrcoef(Returns);

%Visualization
figure(1),subplot(2,2,1),plot((1:T)',Data(:,2),'b-','LineWidth',1);
title('Series 1')  
hold on;
subplot(2,2,2),plot((1:T)',Data(:,3),'b-','LineWidth',1);
title('Series 2')   
hold on;
subplot(2,2,3),plot((1:T)',Data(:,4),'b-','LineWidth',1);
title('Series 3') 
hold on;
subplot(2,2,4),plot((1:T)',Data(:,5),'b-','LineWidth',1);
title('Series 4') 
grid on;