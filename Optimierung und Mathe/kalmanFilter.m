% Ryan Turner (rt324@cam.ac.uk)

% Implements EKF pseudo-code from p. 42 of Thrun, Burgard, and Fox (2005)
% TODO comment inputs and outputs
% INPUTS:
% y(:,t) - the observation at time t
% A - the system matrix
% C - the observation matrix
% Q - the system covariance
% R - the observation covariance
% init_x - the initial state (column) vector
% init_V - the initial state covariance
%
% TODO add options to return one-step predictions in x space or y space
%     - maybe even n step ahead prediction with step argument
%     step = 0 => filter x, step = +n => n step ahead prediction in x
%     step = -n => n step ahead prediction in y.
% add control inputs U
% Should we change some of the covariance matrix operations to work with chol
% factor versions to be really anal about numerical stability?
%
% [m_x, S_x] = kalmanFilter(Y, A, C, Q, R, initx, initV);
% should be give identical results (upto machine prec) to
% [m_x, S_x] = kalman_filter(Y, A, C, Q, R, initx, initV); from Kevin Murphy
function [m_x, S_x, m_y, S_y] = kalmanFilter(Y, A, C, Q, R, initx, initV)

% Get the dimensions of the observed space D and the total time time T
[D T] = size(Y);
% Infer the dimensions of the latent space M
M = size(initx, 1);

% size consistency checks and posdef checks on cov matrices
assert(all(size(A) == [M M]));
assert(all(size(C) == [D M]));
assert(all(size(Q) == [M M]));
assert(all(size(R) == [D D]));
assert(all(size(initx) == [M 1]));
assert(all(size(initV) == [M M]));
assert(isposdef(Q));
assert(isposdef(R));
assert(isposdef(initV));
assert(isKosher(Y)); % TODO assert all params are Kosher too

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
  % Cov[y_t|y_1:t-1].
  % Note that a distribution on x_t must have been performed in the previous
  % iteration. (xp, Cp) are the initial state dist on the 1st iteration.
  [yp, Cy] = predict_y(xp, Cp, C, R); % [D x 1, D x D]
  % incorporate the actual y_t value in our distribution on x_t.
  % m_x = E[x_t|y_1:t], S_x = Cov[x_t|y_1:t]. [M x 1, M x M]
  [m_x(:, t), S_x(:, :, t)] = filter_update(Y(:, t), yp, Cy, xp, Cp, C);

  % Log the one step ahead predictions
  m_y(:, t) = yp; % D x 1
  S_y(:, :, t) = Cy; % D x D

  % Do prediction step:
  % Now we predict x_t+1 in preperation for the next iteration.
  % This line could go at the beginning of the loop, but then we would need a
  % redundant predict_y(), filter_update() for t = 1.
  % xp = E[x_t+1|y_1:t], Cp = Cov[x_t+1|y_1:t]. [M x 1, M x M]
  [xp, Cp] = prediction_step(m_x(:, t), S_x(:, :, t), A, Q);
end

function [xp, Cp] = prediction_step(m_x, S_x, A, Q)
% The mean prediction under the linearization: E[x_t|y_1:t-1]
% Line 2 of Thrun
xp = A * m_x; % M x 1
% The covariance of the prediction under the linearization: Cov[x_t|y_1:t-1]
% Line 3 of Thrun
Cp = A * S_x * A' + Q; % M x M
Cp = (Cp + Cp') / 2;

function [yp, Cy] = predict_y(xp, Cp, C, R)
% The mean prediction under the linearization: E[y_t|y_1:t-1]
yp = C * xp; % D x 1
% The covariance of the prediction under the linearization: Cov[y_t|y_1:t-1]
Cy = C * Cp * C' + R; % D x D
Cy = (Cy + Cy') / 2;

function [m_x, S_x] = filter_update(y, yp, Cy, xp, Cp, C)
% Find the "innovation": difference between point prediction and actual.
innovation = y - yp; % D x 1. = y_t - E[y_t|y_1:t-1]
% Calculate the Kalman gain matrix
% Line 4 of Thrun
% We would use K = Cp * C' * inv(Cy); if we didn't care about numerical
% stability. We use / instead of inv().
K = (Cp * C') / Cy; % M x D
% We update our estimate of x_t using y_t. Line 5 of Thrun
m_x = xp + K * innovation; % M x 1. = E[x_t|y_1:t]
% Update the covariance estimate of x_t using y_t. Line 6 of Thrun
S_x = Cp - K * C * Cp; % M x M. = Cov[x_t|y_1:t]
% TODO factorize into (I - K*C)*Cp? Is this more numerically stable?
S_x = (S_x + S_x') / 2;
