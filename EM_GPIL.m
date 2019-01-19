function [ThetaFinal, Mean_S_1bisT, Cov_S_1bisT, R_S_2bisT] = EM_GPIL(Theta0,thetaDims,Ytrain)

% EM_GPIL Applys the Expectation Maximization algorithm to the GPIL schema,
% Inputs:
%       Theta0 = Initial Gess of the Hyperparameters
%       
%
%
%
%
%
%
%
%
%
%
% Get the dimensions of the observed space D and the total time time T
[D T] = size(Ytrain);
% Infer the dimensions of the latent space M
M = size(initx, 1);

%% Initialization
theta = Theta0;
Error = inf;
%% Expectation Maximization
while Error > 0.1
%% Inference (E-Step)
%% Forward
[loghyperSys, inputSys, targetSys, loghyperObs, inputObs, targetObs, A, C] = unpackFromVector(theta, thetaDims);

[m_x, S_x, m_y, S_y] = gpadf(Ytrain, loghyperSys, inputSys, targetSys, ...
                             loghyperObs, inputObs, targetObs, initial_States_mean, initia_State_Cov, A, C);

%% Backward


%% LEarning (M-Step)
[theta, thetaDims] = packToVector(loghyperSys, inputSys, targetSys, ...
                                     loghyperObs, inputObs, targetObs, A, C);
[theta, nlml] = rt_minimize(theta, 'gpadfdLik', -50, thetaDims, Ytrain);
end

ThetaFinal = 0;
Mean_S_1bisT = 0;
Cov_S_1bisT = 0;
R_S_2bisT = 0;
end