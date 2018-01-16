
function [ OUT, qu ] = QUData( varargin )
% quantize/unquantize data   (by Ron Appel)
%
%  USAGE:
% [QQ,qu] = QUData( XX )                      % quantize data in [0,255]
% [QQ,qu] = QUData( qu, X2, *CLFI )           % quantize more data given qu
% [XX,qu] = QUData( qu, QQ, *CLFI )           % unquantize data given qu
%
% XX:     [(nF) x nX] original data
% QQ:uint8[(nF) x nX] quantized data
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]
                                             %% set params
try    [qu,IN] = deal(varargin{1:2});         % try assigning input args
catch, [qu,IN] = deal([], varargin{1}); end   % catch (assign first call type)
try II = varargin{3}; catch, II = []; end     %  set II (CLF inds) if given
                                             %%
if isempty(qu)                               %% create QU if null
  if isa(IN, 'uint8'), OUT = IN; return; end  % just return if uint8 input
  FIN = IN; FIN(isinf(IN)) = NaN;             % remove Infs for min/max calcul.
  MIN = min(FIN, [],ndims(IN));               % get min
  RNG = max(FIN, [],ndims(IN)) - MIN;         % get range
  R0 = ~(RNG > 0); RNG(R0) = 2;               % find rng <= 0 or NaN; fix to 2
  MIN(R0) = MIN(R0)-1; MIN(isnan(MIN)) = 0;   %  and center mins
  SCA = 254 ./ double(RNG);                   % get scale (256 -2 re: -/+Inf)
  qu = struct('min',MIN, 'sca',SCA);          % create QU
end                                          %%
                                             %% set MIN, SCA, and mf(midfix)
if (sum(size(II)) > 0)                        % if CLF inds given
  II = max(1, abs(II)); sz = size(II);        %  get ftr inds (ignore polarity)
  MIN = reshape(qu.min(II), sz);              %  get indexed thrs
  SCA = reshape(qu.sca(II), sz); mf = 0;      %  fix thr midway: 0 = 0.5 - 0.5
else                                          % else (CLF inds not given)
  MIN = qu.min; SCA = qu.sca; mf = 0.5;       %  set min, sca, midway btwn. thrs
end                                          %%
                                             %% quantize or unquantize
if isa(IN, 'uint8')                           % if should unquantize
  if (all(SCA(:) <= 0) && ~isempty(SCA))
    OUT = IN; return; end                     %  SCA==0 means leave quantized
  OUT = double(IN) -mf;                       %  mid fix
  OUT = bsxfun(@rdivide, OUT, SCA);           %  unscale
  OUT = cast(OUT, 'like',MIN);                %  unquantize
  OUT = bsxfun(@plus, OUT, MIN);              %  unshift
  OUT(IN<=0) = -Inf; OUT(IN>=255) = Inf;      %  fix -/+Infs
else                                          % else (quantize)
  OUT = double(bsxfun(@minus, IN, MIN));      %  shift
  OUT = bsxfun(@times, OUT, SCA);             %  scale
  OUT = uint8(OUT +mf);                       %  mid fix and quantize
  OUT(OUT<=0   & IN>-Inf) = 1;                %  fix small/-Inf vals
  OUT(OUT>=255 & IN< Inf) = 254;              %  fix large/+Inf vals
end                                          %%
