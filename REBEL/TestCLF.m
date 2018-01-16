function [ HH, varargout ] = TestCLF( CLF, NN_, XX )
% classify data XX using (standard/nested) classifer CLF   (by Ron Appel)
%
%  USAGE:
%  HH        = TestCLF(  CLF, *NN, XX )
% [HH,XF,EC] = TestCLF( NCLF, *NN, XX )
%
% CLF:[nCLF], XX:[(nF) x nX], NN:[nN] (intermediate nWLs)
% HH:single[M x nX x nN] (output scores)
% XF:single[nXF x nX] (extended ftrs), EC:int32[nCLF] (evaluation counts)
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]

                                                       %% fix args if needed
if (nargin < 3), XX = NN_; NN_ = Inf; end               % default NN is Inf
if any(diff(NN_) <= 0), error('Invalid NN!'); end       % assert NN ascending
M = getCLF(CLF, 'M'); nX = size(XX, ndims(XX));         % get M and nX
NN = int32(max(0, min(getCLF(CLF, 'nWL'), NN_)));       % clip NN to [0,nWL]

                                                       %% get call type
if (isscalar(CLF) && nargout <= 1)                     %% test flat if scalar
  if (nX > 0), HH = cTestCLF(CLF, NN, XX); end          % if data, test interms

else                                                   %% test nested if NCLF
  nCLF = getCLF(CLF, 'nCLF');                           % get num CLFs in NCLF
  XF = zeros(nCLF *M, nX, 'single');                    % alloc XF table
  EC = zeros(1, nCLF, 'int32');                         % alloc eval'n counts
  if (nX > 0), HH = cTestCLF(CLF, NN, XX, XF, EC); end  % if data, test interms
  varargout = { XF, EC };                               % set output args

end
