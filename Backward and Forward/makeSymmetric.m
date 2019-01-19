% Ryan Turner rt324@cam.ac.uk
% could test if already symmetric but I don't know if that will speed it up

function Y = makeSymmetric(X)

Y = .5 * (X + X');
