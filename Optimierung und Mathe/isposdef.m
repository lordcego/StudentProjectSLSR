% Ryan Turner (rt324@cam.ac.uk)
% Verifies that X is a valid covariance matrix.  Meaning positive (semi)
% definite and symmetric.

function pd = isposdef(X)

[t, err] = cholcov(X);

pd = true;
if err ~= 0
  pd = false;
end
