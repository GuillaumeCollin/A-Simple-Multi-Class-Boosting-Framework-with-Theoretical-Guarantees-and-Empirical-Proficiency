% Copyright 2016 R. Appel, X.P. Burgos-Artizzu, and P. Perona
% Improved Multi-Class Cost-Sensitive Boosting
% via Estimation of the Minimum-Risk Class
% arXiv:1607.03547 [cs.CV]
%
% Main functions:
%   demoREBEL    - quick demo with standard cost on a 3-class task
%   REBEL        - training a REBEL classifier
%   TestCLF      - classify data using a learned REBEL classifier
%  
% Miscellaneous:
%   getCLF       - get CLF subset or properties of a given CLF
%   QUData       - quantize/unquantize data
%   prepareData  - prepare DATA struct for REBEL processing
%   getData      - extract requested data and class sizes from DATA struct
%
% Pre-compiled mex libraries (Windows and Linux 64bit):
%   cREBEL, cTestCLF
%   
% Research paper: REBEL_ArXiv16_Appel.pdf
