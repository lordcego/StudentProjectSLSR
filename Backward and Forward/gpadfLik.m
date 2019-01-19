

function nlml = gpadfLik(theta, thetaDims, Y)

[loghyperSys, inputSys, targetSys, loghyperObs, inputObs, targetObs,A,C]= unpackFromVector(theta, thetaDims);

[D, T] = size(Y);
M = size(loghyperSys, 2);
initx = zeros(M, 1);
initV = eye(M);

[m_x, S_x, m_y, S_y] = gpadf(Y, loghyperSys, inputSys, targetSys, ...
                                loghyperObs, inputObs, targetObs, initx, initV);

nlml = zeros(T, 1);
for t = 1:T
  quadform = (Y(:, t) - m_y(:, t)).' * (S_y(:, :, t) \ (Y(:, t) - m_y(:, t)));
  nlml(t) = 0.5 * quadform + log(sqrt(det(S_y(:, :, t)))) + D * log(2 * pi) / 2;
end

nlml = sum(nlml);
