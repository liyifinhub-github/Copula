% Yi Li June 2022
% ARMA
residuals = nan(T, N);
mean_order = nan(N, 2);

for i=1:N
    [theta,sig,vcv,order,resids1] = ARMAX_opt(Returns(:, i),5,5,'AIC');  
    mean_order(i, 1:2) = order';
    residuals(:, i) = [zeros(max(order), 1);resids1]; % fill in max order1 zeros
end

% EGARCH
Vol = nan(T, N);
EGARCH = egarch('GARCHLags',1:2,'ARCHLags',1:2);
Mdl.Distribution = "Gaussian";
for i= 1: N
    ModelEst = estimate(EGARCH, residuals(:, i));
    voli = infer(ModelEst, residuals(:, i));
    Vol(:, i) = voli;
end

% Retest Standardized Series
stdresids = residuals./sqrt(Vol);

% SKEWT
options = optimset('Display','off','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'DiffMaxChange',Inf,'DiffMinChange',0,'Algorithm','active-set');
outSKEWT = nan(N,2); %dof and asymmetry parameters
lower = [2.1, -0.99];
upper = [Inf, 0.99 ];
theta0 = [6;0];
for i = 1: N
    theta1 = fmincon('skewtdis_LL',theta0,[],[],[],[],lower,upper,[],options,stdresids(:,i));
    outSKEWT(i,:) = theta1';
end

Uskewt = nan(T,N);
for i=1: N
    Uskewt(:, i) = skewtdis_cdf(stdresids(:,i),outSKEWT(i,1),outSKEWT(i,2));
end