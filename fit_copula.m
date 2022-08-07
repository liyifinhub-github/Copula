%Yi Li June 2022
% Copula
tau_tail = nan(5,2);
opt_copula_model= nan(1,N-1);
i=1;
for j=2:N
    u=Uskewt(:,i);
    v=Uskewt(:,j);
   
    options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6);
    % 1. Normal Copula
    kappa1 = corrcoef12(norminv(u),norminv(v));
    LL1 = NormalCopula_CL(kappa1,[u,v]);	   
        
    % 2. Rotated Clayton copula 
    lower = 0.0001;
    theta0 = 1;
    [ kappa2 LL2] = fmincon('claytonCL',theta0,[],[],[],[],lower,[],[],options,1-[u,v]);
        
    % 3. Gumbel copula
    lower = 1.1;
    theta0 = 2;
    [ kappa3 LL3] = fmincon('gumbelCL',theta0,[],[],[],[],lower,[],[],options,[u,v]);
        
    % 4. Student's t copula
    lower = [-0.9 , 2.1 ];
    upper = [ 0.9 , 100 ];
    theta0 = [kappa1;10];
    [ kappa4 LL4] = fmincon('tcopulaCL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);
        
    % 5. Symmetrised Joe-Clayton copula
    lower = [0 , 0 ];
    upper = [ 1 , 1];
    theta0 = [0.25;0.25];
    [ kappa5 LL5] = fmincon('sym_jc_CL',theta0,[],[],[],[],lower,upper,[],options,[u,v]);
        
    LL = [LL1;LL2;LL3;LL4;LL5];
    opt = find(LL==min(LL));
    opt_copula_model(:,j-1) = opt;

    %Tail dependence
    tau_tail(1,:,j-1) = [0,0];                 % Normal copula: zero upper and lower 
    tau_tail(2,:,j-1) = [0,2^(-1/kappa2)];     % Rotated Clayton copula: zero lower 
    tau_tail(3,:,j-1) = [0,2-2^(1/kappa3)];    % Gumbel copula: zero lower 
    tau_tail(4,:,j-1) = ones(1,2)*2*tdis_cdf(-sqrt((kappa4(2)+1)*(1-kappa4(1))/(1+kappa4(1))),kappa4(2)+1);  % Student's t copula: symmetric dependence
    tau_tail(5,:,j-1) = kappa5([2,1])';               % SJC copula parameters: upper and lower 
end