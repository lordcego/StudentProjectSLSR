
function gpplot(loghyper, covfunc, x, y, A)

gpoints = 100;

% A is slope of mean function
if nargin < 5 || isempty(A)
  A = 0;
end
assert(isscalar(A));

xtest = linspace(min(x), max(x), gpoints)';
[mu, S2] = gpr(loghyper, covfunc, x, y, xtest);
errorbar_gpml(xtest, mu + A * xtest, S2);
hold on;
h = stem(x, y, 'LineWidth', 1);
set(get(h,'BaseLine'), 'LineWidth', 1);
set(get(h,'BaseLine'), 'LineStyle', ':');
