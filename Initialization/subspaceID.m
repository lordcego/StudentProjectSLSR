function [A, B, C, D, Q, R, S] = subspaceID(y, u, hankelSize, stateDim)
% x_{t+1} = Ax_t + Bu_t
% y_{t+1} = Cx_{t+1} + Du_t
%
% Q is state covariance
% R is observation covariance
% S is cross covariance
%
% Byron Boots (beb@cs.cmu.edu)

dimU = size(u, 1);
[dimY, ntr] = size(y);

% Build block Hankel matrix
% disp('building block Hankel matrix');
tau = ntr - hankelSize + 1;
Yfp = zeros(hankelSize * dimY, tau);
Ufp = zeros(hankelSize * dimU, tau);
for i = 1:hankelSize
  for j = 1:tau
    Yfp((i - 1) * dimY + 1:i * dimY, j) = y(:, (j - 1) + i);
    Ufp((i - 1) * dimU + 1:i * dimU, j) = u(:, (j - 1) + i);
  end
end

% Remove influence of future actions
% disp('oblique projection 1');
Yf = (Yfp(ceil(end / 2) + 1:end, :));
Yp = (Yfp(1:ceil(end / 2), :));
Uf = (Ufp(ceil(end / 2) + 1:end, :));
Up = (Ufp(1:ceil(end / 2), :));
O = Yf * pinv([Yp; Up; Uf]);
O = O(:, 1:(size(Yp, 1) + size(Up, 1))) * [Yp; Up];

% Find the subspace
% disp('subspace id');
[U2, S2, T2] = svd(O, 'econ');
U2 = double(U2(:, 1:stateDim));
T2 = T2(:, 1:stateDim);
S2 = S2(1:stateDim, 1:stateDim);

% States at time t
state1 = double(S2 * T2');

% Remove influence of future actions
% disp('oblique projection 2');
Ypplus = [Yp; Yf(1:dimY, :)];
Upplus = [Up; Uf(1:dimU, :)];
Yfminus = Yf(dimY + 1:end, :);
Ufminus = Uf(dimU + 1:end, :);
Oplus = Yfminus * pinv([Ypplus; Upplus; Ufminus]);
Oplus = Oplus(:, 1:(size(Ypplus, 1) + size(Upplus, 1))) * [Ypplus; Upplus];

% States at time t+1
state2 = double(pinv(U2(1:end - dimY, :)) * Oplus);

% Unconstrained System
% We can substitute a constraint generation step here to gaurentee
% system stability. See Siddiqi et al. NIPS 2007.
M = [state2; Ypplus(end - dimY + 1:end, :)] * pinv([state1; Uf(1:dimU, :)]);

% The parameters of the dynamical system are submatrices of M
A = M(1:stateDim, 1:stateDim);
B = M(1:stateDim, stateDim + 1:end);
C = M(stateDim + 1:end, 1:stateDim);
D = M(stateDim + 1:end, stateDim + 1:end);

% Errors
E = [state2; Ypplus(end - dimY + 1:end, :)] - M * [state1; Uf(1:dimU, :)];
E = (1 / (ntr - 1)) * (E * E');

% Covariances
Q = E(1:stateDim, 1:stateDim);
S = E(1:stateDim, stateDim + 1:end);
R = E(stateDim + 1:end, stateDim + 1:end);
