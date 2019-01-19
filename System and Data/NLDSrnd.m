

function [Y, X] = NLDSrnd(T, ffunc, gfunc, Q, R, initx, initV)

% Get the dimensions of the observed space D and the total time time T
D = size(R, 1);
% Infer the dimensions of the latent space M
M = size(initx, 1);

% size consistency checks and posdef checks on cov matrices
assert(all(size(Q) == [M M]));
assert(all(size(R) == [D D]));
assert(all(size(initx) == [M 1]));
assert(all(size(initV) == [M M]));
assert(isposdef(Q));
assert(isposdef(R));
assert(isposdef(initV));
% TODO assert all params are Kosher too
% TODO check size outputs of functions

% Pre-load the result matrices
X = zeros(M, T);
Y = zeros(D, T);

X(:, 1) = mvnrnd(initx, initV)';
Y(:, 1) = gfunc(X(:, 1)) + mvnrnd(zeros(M, 1), R)';
for ii = 2:T
  X(:, ii) = ffunc(X(:, ii - 1)) + mvnrnd(zeros(M, 1), Q)';
  Y(:, ii) = gfunc(X(:, ii)) + mvnrnd(zeros(M, 1), R)';
end
