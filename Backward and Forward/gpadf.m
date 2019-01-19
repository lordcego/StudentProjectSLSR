% Ryan Turner (rt324@cam.ac.uk)

% Loop is closely adapted from algorithm in:
% http://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
% Also implements UKF pseudo-code from p. 70 of Thrun, Burgard, and Fox (2005)
%
% Input:
% Y   D x T matrix of observations
% ffunc   function handle with black box model of state transitions
%         ffunc(X), X is M x N, must return M x N matrix with state transition
%         for each col in X.
% gfunc   function handle with black box model of measurement model
%         gfunc(X), X is M x N, must return D x N matrix with measurement
%         for each col in X.
% Q   M x M pos def matrix of system noise = Cov[X_t|X_t-1]
% R   D x D pos def matrix of measurement noise = Cov[Y_t|X_t]
% initx   M x 1 vector for mean of initial state distribution = E[X_1]
% initV   M x M pos def matrix for cov of inital state distribution = Cov[X_1]
%
% Output:
% m_x   M x T mean of filtered state. m_x(i,t) = E[X(i,t)|Y(:,1:t)]
% S_x   M x M x T covariance of filterred state. S_x(:,:,t) =
% Cov[X(:,t)|Y(:,1:t)]
% m_y   D x T mean of one step ahead predictions. m_y(i,t) =
% E[Y(i,t)|Y(:,1:t-1)]
% S_y   D x D x T covariance of one step ahead predictions. S_y(:,:,t) =
% Cov[Y(:,t)|Y(:,1:t-1)]
%
% TODO add options to return one-step predictions in x space or y space
%     - maybe even n step ahead prediction with step argument
%     step = 0 => filter x, step = +n => n step ahead prediction in x
%     step = -n => n step ahead prediction in y.
% add control inputs U
% add option to not resample
% Should we change some of the covariance matrix operations to work with chol
% factor versions to be really anal about numerical stability?
function [m_x, S_x, m_y, S_y] = gpadf(Y, loghyperSys, inputSys, targetSys, ...
  loghyperObs, inputObs, targetObs, initx, initV)

% Get the dimensions of the observed space D and the total time time T
[D T] = size(Y);
% Infer the dimensions of the latent space M
M = size(initx, 1);

% Pre-load the result matrices
m_x = zeros(M, T);
S_x = zeros(M, M, T);
m_y = zeros(D, T);
S_y = zeros(D, D, T);

% This is the initial next step ahead on x.
% p(x_1|no y) = init state distribution.
xp = initx; % M x 1
Cp = initV; % M x M

% The loop order here has been permuted from what is standard, but it eliminates
% the need for redundant code to hande the t = 1 iteration.
% It is usually: prediction_step(), predict_y(), filter_update()
% but to make initialization more natural the loop if now:
% predict_y(), filter_update(), prediction_step().
for t = 1:T
  % Do the update step:
  % predict y_t as in p(y_t|y_1:t-1). yp is E[y_t|y_1:t-1]. Cy is
  % Cov[y_t|y_1:t-1]. crossSigma is Cov[x_t, y_t|y_1:t-1].
  % Note that a distribution on x_t must have been performed in the previous
  % iteration. (xp, Cp) are the initial state dist on the 1st iteration.
  % [D x 1, D x D, M x D]
  [yp, Cy, crossSigma] = gpur(loghyperObs, inputObs, targetObs, xp, Cp);
  yp = yp';
  % Incorporate the actual y_t value in our distribution on x_t.
  % m_x = E[x_t|y_1:t], S_x = Cov[x_t|y_1:t]. [M x 1, M x M]
  [m_x(:, t), S_x(:, :, t)] = ...
    filter_update(Y(:, t), yp, Cy, xp, Cp, crossSigma);

  % Log the one step ahead predictions
  m_y(:, t) = yp; % D x 1
  S_y(:, :, t) = Cy; % D x D

  % Do prediction step:
  % Now we predict x_t+1 in preperation for the next iteration.
  % This line could go at the beginning of the loop, but then we would need a
  % redundant predict_y(), filter_update() for t = 1.
  % xp = E[x_t+1|y_1:t], Cp = Cov[x_t+1|y_1:t]. [M x 1, M x M, M x nPts]
  [xp, Cp] = gpur(loghyperSys, inputSys, targetSys,  m_x(:, t), S_x(:, :, t));
  xp = xp';
end

function [m_x, S_x] = filter_update(y, yp, Cy, xp, Cp, crossSigma)
% Find the "innovation": difference between point prediction and actual.
innovation = y - yp; % D x 1. = y_t - E[y_t|y_1:t-1]
% Calculate the Kalman gain matrix
% Line 11 of Thrun
% We would use K = crossSigma * inv(Cy); if we didn't care about numerical
% stability. We use / instead of inv().
K = crossSigma / Cy; % M x D
% We update our estimate of x_t using y_t. Line 12 of Thrun
m_x = xp + K * innovation; % M x 1. = E[x_t|y_1:t]
% Update the covariance estimate of x_t using y_t. Line 13 of Thrun
S_x = Cp - K * Cy * K'; % M x M. = Cov[x_t|y_1:t]
S_x = makeSymmetric(S_x); % M x M

assert(isposdef(S_x));
