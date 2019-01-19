% Ryan Turner rt324@cam.ac.uk
% inputs must have dims <= 2 for now
% TODO extend to 3+

function [packed dims] = packToVector(varargin)

dims = zeros(nargin, 2);
% TODO setup way to pre-load
packed = [];

for i = 1:nargin
  packed = [packed; varargin{i}(:)];
  dims(i, :) = size(varargin{i});
end
