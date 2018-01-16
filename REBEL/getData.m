
function varargout = getData( DATA, rvs_ )
% extract requested data and class sizes from DATA struct   (by Ron Appel)
%
%  USAGE:
% [X1,NY1, ...] = getData( DATA, *rvs );
%
%  IN/OUTPUTS
% DATA: data struct (see below)
% rvs: groupings i.e. ['r v s'], 'rv s', 'rvs', 'r s', 'x'
%
% DATA:
%  .XR:[(nF) x nXR], .NYR:[M]  % training data subset and class sizes
%  .XV:[(nF) x nXV], .NYV:[M]  % validation data subset and class sizes
%  .XS:[(nF) x nXS], .NYS:[M]  % test data subset and class sizes
% -or-
%  .XX:[(nF) x nX],  .NY:[M]   % consolidated (full) dataset and class szs
%                              % most functions don't use .XX/.NY
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]

%% resolve optional arg rvs
if (nargin < 2 || isempty(rvs_)), rvs_ = 'r v s'; end

%% remove empty subdata from rvs
for x = 'xrvs'
  if ~isfield(DATA, ['X' upper(x)]), rvs_(rvs_==x) = '_'; end
end

%% extract all groups in rvs, append to output
G = textscan(rvs_, '%s'); G = G{1}; nG = numel(G);
varargout = cell(1, max(nargout, 2*nG));
for g = 1:nG, [varargout{2*g-[1 0]}] = catGroup(G{g}); end

%% concatenate requested subgroups into one group
function [ XX, NY ] = catGroup( grp )
  % get group data
  if  any(grp == 'x'),       XX = DATA.XX; NY = DATA.NY; return; end
  r = any(grp == 'r'); if r, XX = DATA.XR; NY = DATA.NYR; end
  v = any(grp == 'v'); if v, XX = DATA.XV; NY = DATA.NYV; end
  s = any(grp == 's'); if s, XX = DATA.XS; NY = DATA.NYS; end
  z = r+v+s;
  if (z <= 1),  if (z <= 0), XX = []; NY = int32([]); end; return;  end

  % init outputs and get size
  M = numel(NY); sz = size(XX); sz(end) = []; nF = prod(sz);
  XM = cell(1, M); NY = zeros(1, M, 'int32');

  % concat data (reshape to original shape)
  if r, catData('R'); end
  if v, catData('V'); end
  if s, catData('S'); end
  XX = [XM{:}]; XX = reshape(XX, [sz size(XX, 2)]);
  
  function catData( x )
    XX = DATA.(['X' x]); NYD = DATA.(['NY' x]);
    XX = reshape(XX, [nF numel(XX)/nF]); NY = NY + NYD;
    I = 0; for m = 1:M, I = I(end)+(1:NYD(m)); XM{m} = [XM{m} XX(:,I)]; end
  end
  
end

end
