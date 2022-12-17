

tab = table2array(readtable("HestonGreeks5000.csv"));

rho = tab(:,3);
sigma = tab(:,4);
kappa = tab(:,5);
nu = tab(:,6);
r = tab(:,7);
K = tab(:,9);
V = tab(:,10);
Settle = 0;
Maturity = tab(:,8).*365;


n = length(V);
theta = zeros(n,1);
delta = zeros(n,1);
vegalt = zeros(n,1);
vega = zeros(n,1);
rho = zeros(n,1);

for j = 1:n
j
theta(j) = optSensByHestonFFT(r(j),1,Settle,Maturity(j),"call",K(j),V(j),nu(j),kappa(j),sigma(j),rho(j), 'DividendYield', 0, 'OutSpec', "theta" );
delta(j) = optSensByHestonFFT(r(j),1,Settle,Maturity(j),"call",K(j),V(j),nu(j),kappa(j),sigma(j),rho(j), 'DividendYield', 0, 'OutSpec', "delta" );
vegalt(j) = optSensByHestonFFT(r(j),1,Settle,Maturity(j),"call",K(j),V(j),nu(j),kappa(j),sigma(j),rho(j), 'DividendYield', 0, 'OutSpec', "vegalt" );
vega(j) = optSensByHestonFFT(r(j),1,Settle,Maturity(j),"call",K(j),V(j),nu(j),kappa(j),sigma(j),rho(j), 'DividendYield', 0, 'OutSpec', "vega" );
rho(j) = optSensByHestonFFT(r(j),1,Settle,Maturity(j),"call",K(j),V(j),nu(j),kappa(j),sigma(j),rho(j), 'DividendYield', 0, 'OutSpec', "rho" );

end


greeks_ = [theta delta vega vegalt rho];
csvwrite('Greeks5000.csv',greeks_);



