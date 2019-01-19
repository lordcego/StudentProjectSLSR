% Ryan Turner rt324@cam.ac.uk
% inputs must have dims <= 2 for now
% TODO extend to 3+

function varargout = unpackFromVector(packed, dims)

% assert all the dimensions are consistent
[p_rows p_cols] = size(packed);
assert(p_cols == 1);
total_out = sum(prod(dims, 2));
assert(p_rows == total_out);
assert(size(dims, 2) == 2);

numout = size(dims, 1);
assert(nargout == numout);

end_idx = 0;
for i = 1:numout
  start_idx = end_idx + 1;
  end_idx = start_idx + prod(dims(i, :)) - 1;
  varargout{i} = reshape(packed(start_idx:end_idx), dims(i, 1), dims(i, 2));
end
