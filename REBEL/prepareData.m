function [DATA,tr,valid,test]=prepareData(X,Y,ratios)
% Preparing data in REBEL format 
%
%  USAGE:
% [DATA,tr,valid,test]=prepareData(X,Y,ratios)
%
%  IN/OUTPUTS:
% DATA: data struct (type: help getData), CC:double[M x K]
% pTrain: .depth, .nWL (tree depth and num weak learners, -nWL: skip A0)
%         [.WW -or- .CLF] (initial weights or CLF)
%         [.F], [.lazy] (ftr subset, fraction)
%         [.log] (log file)
%
% CLF: classifier
%
% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]
[N,D]=size(X); numClasses=max(Y);

NTr=round(N*ratios(1));NValid=round(N*ratios(2)); NTest=N-NTr-NValid;

all_ex=1:N; tr=randsample(all_ex,NTr); 
all_ex=all_ex(~ismember(all_ex,tr));valid=randsample(all_ex,NValid); 
test=all_ex(~ismember(all_ex,valid));

NYS=zeros(1,numClasses);NYR=zeros(1,numClasses);NYV=zeros(1,numClasses);
XS=zeros(D,NTest);ks=1;
XR=zeros(D,length(tr));kr=1;
XV=zeros(D,length(valid));kv=1;
for cl=1:numClasses
    ind=find(Y==cl); 
    
    indR=ind(ismember(ind,tr));nExR=length(indR); 
    indV=ind(ismember(ind,valid));nExV=length(indV); 
    indS=ind(ismember(ind,test));nExS=length(indS); 
    
    NYR(cl)=nExR;NYV(cl)=nExV;NYS(cl)=nExS;
    
    XR(:,kr:kr+nExR-1)=X(indR,:)'; kr=kr+nExR;
    XV(:,kv:kv+nExV-1)=X(indV,:)'; kv=kv+nExV;
    XS(:,ks:ks+nExS-1)=X(indS,:)'; ks=ks+nExS;
end

DATA.XS=single(XS);DATA.XR=single(XR);DATA.XV=single(XV);
DATA.NYS=int32(NYS);DATA.NYR=int32(NYR);DATA.NYV=int32(NYV);
end
