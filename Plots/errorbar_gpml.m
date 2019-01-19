

% We use 95% intervals here.
% Note that S is the variance not the errorbar size
function errorbar_gpml(xstar, mu, S)

% TODO sort xstar

% They all need to be col vecs for this to work
xstar = xstar(:);
mu = mu(:);
S = S(:);

region_color = [7 7 7] / 8;
edge_color = [7 7 7] / 8;

hold_state = ishold;

% TODO explain this
f = [mu + 1.96 * sqrt(S); flipdim(mu - 1.96 * sqrt(S), 1)];
fill([xstar; flipdim(xstar, 1)], f, region_color, 'EdgeColor', edge_color);
hold on;
plot(xstar, mu, 'k-', 'LineWidth', 2);
axis tight;

if ~hold_state
  hold off;
end
