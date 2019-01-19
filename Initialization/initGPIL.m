

function [theta0, thetaDims] = initGPIL(Y, M, N)
% Initialize GPIL algo:
%  Y D x T
%  M is 1 x 1 is latent dim
%  N is 1 x 1 number of pseuo-points

useLinearMean = false;

[D, T] = size(Y);

maxIter = 100;
hankelSize = 16;

% Initial state of latent state
initx = zeros(M, 1);
initV = eye(M);

% Make sure the settings and data are sane
assert(N >= 1);
assert(M >= 1);
assert(D >= 1);
assert(N <= T);

% Get hard estimates for latent states:
% Use subspace ID to find the optimal Kalman filters params with no local opt.
% [M x M, D x M, M x M, D x D]
[A, tmdp, C, tmp, Q, R] = subspaceID(Y, zeros(0, T), hankelSize, M);
% Now find the latent states with Kalman filter. We might want to bump this up
% to Kalman smoother. M x T
m_x = kalmanFilter(Y, A, C, Q, R, initx, initV);
% Standardize the results since latent state scale is arbitrary. We could go
% furhter and do whitening or transform so that Q is diag. M x T
% Do robust standardize??
if ~useLinearMean
  % Don't standardize if we are using the linear mean since A and C are scaled
  % towards the orignal m_x.
  m_x = standardize_rows(m_x);
  % Initialize and A and C at 0 instead of N4SID solution.
  A = zeros(M, M);
  C = zeros(D, M);
end

% Now that we have hard estimates we can get parameters without the difficulty
% of uncertain inputs. Since GPIL will use sparse we will train a sparse GP.
% Initialize the pseudo-inputs with random sub sample of the data.
tsample = randsample(T, N); % N x 1
inputSys = m_x(:, tsample)'; % N x M
inputObs = inputSys; % N x M

trainingInput = m_x(:, 1:end - 1);
trainingOutput = m_x(:, 2:end) - A * m_x(:, 1:end - 1);

% Sparse learning of pseudo-inputs will be much more robust for fixed hypers. =>
% we find the hypers using non-sparse GP.
covfunc = {'covSum', {'covSEard', 'covNoise'}};
% Sane initialization of GP hypers given scale of m_x
loghyperSys = repmat(log([ones(1, M) 1.0 0.1]'), 1, M); % M + 2 x M
for ii = 1:M
  % Must learn the hypers one output dim at a time. M + 2 x 1
  loghyperSys(:, ii) = rt_minimize(loghyperSys(:, ii), 'gpr', -maxIter, ...
    covfunc, trainingInput', trainingOutput(ii, :)');
end
% Now that we have the hypers we learn the sparse inputs. NM x 1
inputSys = rt_minimize(inputSys(:), @spgpWrap, -maxIter, loghyperSys, ...
  trainingInput', trainingOutput', N);
% Reshape to the correct size. N x M
inputSys = reshape(inputSys, N, M);

% Since there are no uncertain inputs => we only had to optimize the inputs =>
% we now call SPGP to use posterior mean to get pseuo-targets. Better since did
% not require optimization of outputs => less chance to overfit.
targetSys = zeros(N, M);
for ii = 1:M
  % Must do each dimension independently. N x 1
  targetSys(:, ii) = spgp_pred(trainingOutput(ii, :)', trainingInput', ...
    inputSys, inputSys, loghyperSys(:, ii));
end

% Now we must do it over again for the observation model.
trainingInput = m_x;
trainingOutput = Y - C * m_x;

% Sane initialization of GP hypers given scale of m_x
loghyperObs = repmat(log([ones(1, M) 1.0 0.1]'), 1, D); % M + 2 x D
for ii = 1:D
  % Must learn the hypers one output dim at a time. M + 2 x 1
  loghyperObs(:, ii) = rt_minimize(loghyperObs(:, ii), 'gpr', -maxIter, ...
    covfunc, trainingInput', trainingOutput(ii, :)');
end
% Learn the sparse inputs. NM x 1
inputObs = rt_minimize(inputObs(:), @spgpWrap, -maxIter, ...
  loghyperObs, trainingInput', trainingOutput', N);
% Put back in correct shape. N x M
inputObs = reshape(inputObs, N, M);

% Get pseudo-targets using posterior mean.
targetObs = zeros(N, D);
for ii = 1:D
  % Go over each output dimension. N x 1
  targetObs(:, ii) = spgp_pred(trainingOutput(ii, :)', trainingInput', ...
    inputObs, inputObs, loghyperObs(:, ii));
end

% Pack all of this into a vector for further learning. col vector.
[theta0, thetaDims] = packToVector(loghyperSys, inputSys, targetSys, ...
  loghyperObs, inputObs, targetObs, A, C);

function [nlml, dnlml] = spgpWrap(xb, loghyper, X, Y, N)
% xb ND x 1 = sparseInputs(:) (N x D)
% loghyper (D + 2) x E SE-ARD hypers for each output dimension
% X N2 x D function inputs
% Y N2 x E function outputs
% N 1 x 1 number of sparse inputs

E = size(Y, 2);

nlml = 0;
dnlml = zeros(size(xb, 1), 1);
for ii = 1:E
  w = [xb; loghyper(:, ii)];
  [fw, dfw] = spgp_lik_nohyp(w, Y(:, ii), X, N);
  nlml = nlml + fw;
  dnlml = dnlml + dfw(1:size(xb, 1));
end
