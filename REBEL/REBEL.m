function CLF = REBEL( DATA, CC_, pTrn )
% REBEL using trees with validation   (by Ron Appel)
%
%  USAGE:
% CLF = REBEL( DATA, [CC], pTrain );
%
%  IN/OUTPUTS:
% DATA: data struct (type: help getData), CC:double[M x K]
% pTrain: .depth, .nWL (tree depth and num weak learners, -nWL: skip A0)
%         [.WW -or- .CLF] (initial weights or CLF)
%         [.F], [.lazy] (ftr subset, fraction)
%         [.log] (log file)
%
% CLF: REBEL classifier
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]
                                                         %% parse input params
if isstruct(CC_), pTrn = CC_; CC_ = []; end               % parse 2-input call
try WW  = pTrn.WW ; catch, WW  = []; end                  % get wts if given
try CLF = pTrn.CLF; catch, CLF = []; end                  % get clf if given

                                                         %% get data subsets
[XR,NYR] = getData(DATA, 'r');                            % get training data
[QR,qu] = QUData(XR);                                     % quantize

                                                         %% prepare cost matrix
if isempty(CC_), CC_ = 1 - eye(numel(NYR));               % set [1-I] if empty
elseif isvector(CC_), CC_ = 1 - eye(CC_); end             % or parse as [M K]

                                                         %% init train wts WW
pTrn.fixW = isempty(WW);                                  % if WW, ignore CLF
if pTrn.fixW                                              % if wts not given
  try                                                     %  assume CLF given
    WW = double(TestCLF(CLF, XR));                        %   test orig. CLF
    pTrn.nWL = min(0, getCLF(CLF,'nWL') - abs(pTrn.nWL)); %   fix nWL accrdngly
  catch, WW = zeros(size(CC_, 2), size(XR, 2)); end       %  alloc wts (zeros)
end
                                                         %% train
CLF = cREBEL(QR, NYR, CC_, WW, pTrn);                     % invoke cREBEL
CLF.T = QUData(qu, CLF.T, CLF.I);                         % unquantize thrs
