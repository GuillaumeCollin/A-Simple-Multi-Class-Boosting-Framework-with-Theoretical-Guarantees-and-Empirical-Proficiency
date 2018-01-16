function out = getCLF( varargin )
% get CLF subset or properties of a given CLF
%
%  USAGE:
%  NCLF0 = getCLF( nCLF )               % get empty NCLF struct (.A0,.A,.I,.T)
% subCLF = getCLF( CLF, indices )       % get subset of weak learners in CLF
%  value = getCLF( CLF, 'property' )    % get property value of (nested) CLF
%
%  indices: similar to array indexing
%           negative indices: -1 = end, -2 = end-1, etc.
%
%  property:
%     M:  number of classes (M = 1 for binary CLFs)
%   nWL:  number of weak learners in full CLF
%  -nWL:  number of weak learners in full CLF (encode A0 as +/-)
%  nCLF:  number of nested CLFs in NCLF
%     N:  number of: WLs in flat CLF -or- nested CLFs in NCLF
% depth:  depth of trees
% hasA0:  does CLF have A0
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]

if (nargin <= 1)
  S = single([]); out = struct('A0',S, 'A',S, 'I',int32([]), 'T',S);
  if (nargin >= 1), out = repmat(out, [1 varargin{1}]); end

else
  [CLF,arg] = deal(varargin{:});

  if isnumeric(arg)
    if ~isscalar(CLF), error('CLF is not flat!'); end

    zers = arg == 0; arg(zers) = []; negs = arg < 0;
    if any(zers), A0 = getCLF(CLF, 'A0'); else A0 = single([]); end
    if any(negs), arg(negs) = arg(negs) + (getCLF(CLF, 'N') +1); end
    out = struct('A0',A0,'A',CLF.A(:,arg),'I',CLF.I(:,arg),'T',CLF.T(:,arg));
    
  else
    switch arg
      case 'hasA0'
        out = isfield(CLF, 'A0') && ~isempty([CLF.A0]);
      case 'A0'
        if getCLF(CLF, 'hasA0'), out = [CLF.A0]; else out = single(0); end
      case 'M'
        if isempty(CLF), out = 0;
        else
          if getCLF(CLF, 'hasA0'), A = 'A0'; else A = 'A'; end
          out = size(CLF(1).(A), 1);
        end
      case 'depth'
        if isempty(CLF), out = 0;
        else out = log2(size(CLF(1).I, 1) +1); end
      case 'nWL'
        if isempty(CLF), out = 0;
        else out = size([CLF.I], 2); end
      case 'nCLF'
        out = numel(CLF);
      case '-nWL'
        out = getCLF(CLF, 'nWL');
        if (getCLF(CLF, 'hasA0')), out = -out; end
      case 'N'
        if isscalar(CLF), N = 'nWL'; else N = 'nCLF'; end
        out = getCLF(CLF, N);
      otherwise
        error('Invalid CLF property: ''%s''!', arg);
    end
  end
end
