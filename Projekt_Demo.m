clear
clc

%% System Beschreibung
% Noise
SystemVarianz = 0.1;
MeassurementVarianz = 0.1;
Q = SystemVarianz ^ 2;
R = MeassurementVarianz ^ 2;
% System Dynamics
% ffunc = @(x,u,n,t) -x-3*x.^2; 
ffunc = @(x,u,n,t) 3 * sin(3*x);
M = 1;                                                                      % Dimension of the Space
gfunc = @(x,u,n,t) x;
D = 1;                                                                      % Dimension of the Meassurement
% Statistics of the System
initial_States_mean = 0;
initia_State_Cov = 1;
% Computational Dimension
N = 10;                                                                     % Size of the Pseudo set Training (for Hyperparameters)
T = 100;                                                                    % Size of the Training Data for the Model
T_test = 300;                                                               % Size of the test Observation for Errors
% Covariance functions 
covfunc = {'covSum', {'covSEard', 'covNoise'}};
%% Generate data
Ytrain = NLDSrnd(T, ffunc, gfunc, Q, R, initial_States_mean, initia_State_Cov);
Ytest = NLDSrnd(T_test, ffunc, gfunc, Q, R, initial_States_mean, initia_State_Cov);
Ytrain = [Ytrain; randn(D - 1, T)];
Ytest = [Ytest; randn(D - 1, T_test)];

% Statistics of the Data
 meanY_T = mean(Ytrain,2);
 covY_T = cov(Ytrain');
 
%% GPIL initialization
[theta0, thetaDims] = initGPIL(Ytrain, M, N);                               % Initialization of the Hyperparameter with Non-Sparse GP

%% Expectation MAximization
% First Unpack the Hyperparameters out of Initialization Output

[theta, nlml] = rt_minimize(theta0, 'gpadfdLik', -10, thetaDims, Ytrain);

[loghyperSys, inputSys, targetSys, loghyperObs, inputObs, targetObs, A, C] = unpackFromVector(theta, thetaDims);

[m_x, S_x, m_y, S_y] = gpadf(Ytrain, loghyperSys, inputSys, targetSys, ...
                             loghyperObs, inputObs, targetObs, initial_States_mean, initia_State_Cov);
                         
                         
%%      Errors  
% First Initialization of some Variabes for the Errors for latter
models = {'GPIL'};
NLL = zeros(T_test, length(models));
sqError = zeros(T_test, length(models));
absError = zeros(T_test, length(models));
[NLL(:, 1), sqError(:, 1), absError(:, 1)] = getError_innFunction(Ytest, m_y, S_y);


% [ThetaFinal, Mean_S_1bisT, Cov_S_1bisT, R_S_2bisT] = EM_GPIL();
%% Plot

 figure;
 gpplot(loghyperSys, covfunc, inputSys, targetSys, A);
 hold on
 scatter(Ytrain(1:end-1),Ytrain (2:end))
  xlim1 = xlim;
  xstar = linspace(xlim1(1), xlim1(2), 100)';
  
  % Find a warping of the latent space
  % TODO actually find a wraping properly
  gpmean = gpr(loghyperSys, covfunc, inputSys, targetSys, xstar);
  params = regress(gpmean + A, [ones(100,1) ffunc(xstar)]);
  
  f2 = params(2) * ffunc(xstar) + params(1);
  plot(xstar, f2, 'r');
  plot(xstar, f2 + 1.96 * sqrt(Q), 'r--');
  plot(xstar, f2 - 1.96 * sqrt(Q), 'r--');

%% Functionen die in Script gebrauch wurden
function [NLL, sqError, absError] = getError_innFunction(Y, m_y, S_y)

NLL = -mvnlogpdf(Y', m_y', S_y);
sqError = sum((Y - m_y) .^ 2, 1);
absError = sum(abs((Y - m_y)), 1);
end
















