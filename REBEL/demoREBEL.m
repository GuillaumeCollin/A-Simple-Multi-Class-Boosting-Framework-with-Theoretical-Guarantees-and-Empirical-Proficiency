% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]

%% Load data
load fisheriris
X = meas; [classNames,~,Y]=unique(species);
numCl=length(classNames);

%% Separate into train/validation/test set
ratios=[.7 .15 .15];%Train,Valid,Test ratios
[DATA,tr,valid,test]=prepareData(X,Y,ratios);

%% Train REBEL
%Training parameters (tree depth and number of learners)
pTrain = struct('depth',1, 'nWL',50);
%Cost matrix (uniform)
CC_ = 1 - eye(numCl);
%Perform actual training
CLF = REBEL(DATA, CC_, pTrain);

%% Apply learned Rebel classifier to get output confidence scores
confS = TestCLF(CLF,single(X(test,:))');
confV = TestCLF(CLF,single(X(valid,:))');
confR = TestCLF(CLF,single(X(tr,:))');
%get argmax class (final prediction)
[~,hR]=max(confR);[~,hV]=max(confV);[~,hS]=max(confS); 

%% Plot results
NR=length(hR);NV=length(hV);NS=length(hS);
targetR=zeros(numCl,NR);targetV=zeros(numCl,NV);targetS=zeros(numCl,NS);
outR=zeros(numCl,NR);outV=zeros(numCl,NV);outS=zeros(numCl,NS);
for cl=1:numCl
    targetR(cl,Y(tr)==cl)=1;  targetV(cl,Y(valid)==cl)=1; targetS(cl,Y(test)==cl)=1;
    outR(cl,hR==cl)=1; outV(cl,hV==cl)=1; outS(cl,hS==cl)=1; 
end
classNames1={classNames{:},'ALL'};
close all,
figure(1),clf,plotconfusion(targetR,outR)
set(gca,'YTickLabel',classNames1)
set(gca,'XTickLabel',classNames1)
title('TRAIN'),drawnow

figure(2),clf,plotconfusion(targetV,outV)
set(gca,'YTickLabel',classNames1)
set(gca,'XTickLabel',classNames1)
title('VALIDATION'),drawnow

figure(3),clf,plotconfusion(targetS,outS)
set(gca,'YTickLabel',classNames1)
set(gca,'XTickLabel',classNames1)
title('TEST')

fprintf('----------demoREBEL results-------------------\n');
fprintf('Train accuracy %0.3f\n',nnz(hR==Y(tr)')/numel(tr))
fprintf('Validation accuracy %0.3f\n',nnz(hV==Y(valid)')/numel(valid))
fprintf('Test accuracy %0.3f\n',nnz(hS==Y(test)')/numel(test))
fprintf('----------------------------------------------\n');
