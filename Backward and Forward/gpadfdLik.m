

function [nlml, dnlml] = gpadfdLik(theta, thetaDims, Y)

assert(isreal(theta));

epsilon = 1e-15;

disp('.');

nlml = gpadfLik(theta, thetaDims, Y);
dnlml = zeros(size(theta, 1), 1);
for ii = 1:size(theta, 1)
  thetaii = theta;
  thetaii(ii) = thetaii(ii) + epsilon * 1i;
  nlmlii = gpadfLik(real(thetaii), thetaDims, Y);
  dnlml(ii) = imag(nlmlii) / epsilon;
end

assert(isreal(nlml));
assert(isreal(dnlml));
