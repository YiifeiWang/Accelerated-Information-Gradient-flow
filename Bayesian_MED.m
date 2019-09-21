% The main part of this code is based on 
% https://github.com/DartML/Stein-Variational-Gradient-Descent
% Qiang Liu and Dilin Wang. 
% Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm. NIPS, 2016.

addpath('./utils');
addpath('./solvers');

dataset_name = 'covtype';
id_test = 1;

num_repeat = 10;

M = 100; % number of particles

% we partition the data into 80% for training and 20% for testing
train_ratio = 0.8;
max_iter = 6000;  % maximum iteration times

% build up training and testing dataset
load ../data/covertype.mat;
X = covtype(:,2:end); y = covtype(:,1); y(y==2) = -1;

X = [X, ones(size(X,1),1)];  % the bias parameter is absorbed by including 1 as an entry in x
[N, d] = size(X); D = d+1; % w and alpha (prameters)

% building training and testing dataset
train_idx = randperm(N, round(train_ratio*N));  test_idx = setdiff(1:N, train_idx);
X_train = X(train_idx, :); y_train = y(train_idx);
X_test = X(test_idx, :); y_test = y(test_idx);

n_train = length(train_idx); n_test = length(test_idx);

% example of bayesian logistic regression
batchsize = 50; % subsampled mini-batch size
a0 = 1; b0 = .01; % hyper-parameters


dlog_p  = @(theta)dlog_p_Bayesian(theta, X_train, y_train,batchsize);  % returns the first order derivative of the posterior distribution 


SVGD.name = 'SVGD';
SVGD.Pname = 'SVGD';
SVGD.opts = struct('tau',0.05,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
	'itPrint',1e2,'X_test',X_test,'y_test',y_test,'adagrad', 1);

WNAG.name = 'WNAG';
WNAG.Pname = 'WNAG';
WNAG.opts = struct('tau',1e-6,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
		'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9, 'alpha',3.9,'ttype',2);

Wnes.name = 'WNAG';
Wnes.Pname = 'WNes';
Wnes.opts = struct('tau',1e-5,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
		'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9,'wtype',2,'ttype',2);

MCMC.name = 'MCMC';
MCMC.Pname = 'MCMC';
MCMC.opts = struct('tau',1e-5,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',0,...
	'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9);

WGF.name = 'WGF';
WGF.Pname = 'WGF';
WGF.opts = struct('tau',1e-5,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
	'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9,'ttype',1);

AIG.name = 'AIG-SE';
AIG.Pname = 'AIG';
AIG.opts = struct('tau',1e-5,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
	'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9, 'restart',0,'strong',0,'ttype',2);

AIGr.name = 'AIG-SE';
AIGr.Pname = 'AIG-r';
AIGr.opts = struct('tau',1e-6,'iter_num',6e3,'record',1,'trace',1,'ibw',-1,'ktype',1,...
	'itPrint',1e2,'X_test',X_test,'y_test',y_test,'tau_itv', 1e2, 'tau_dec',0.9, 'restart',1,'strong',0,'ttype',1);

algorithm = {MCMC,SVGD,WNAG,Wnes,WGF,AIG,AIGr};
results = {};

for j = 1:num_repeat
	% initlization for particles using the prior distribution
	alpha0 = gamrnd(a0, b0, M, 1); theta0 = zeros(M, D);
	for i = 1:M
	    theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
	end
	alphaMC = gamrnd(a0, b0, (M*10), 1); thetaMC = zeros((M*10), D);
	for i = 1:(M*10)
	    thetaMC(i,:) = [normrnd(0, sqrt((1/alphaMC(i))), 1, d), log(alphaMC(i))]; % w and log(alpha)
	end
	for i = 1:length(algorithm)
		fprintf('Test %d/%d, Alg: %s\n',j,num_repeat,algorithm{i}.Pname);
		switch algorithm{i}.name
			case 'WGF'
				[~,out] = WGF_m(theta0,dlog_p,algorithm{i}.opts);
			case 'SVGD'
				[~,out] = SVGD_m(theta0,dlog_p,algorithm{i}.opts);
			case 'AIG-SE'
				[~,out] = AIG_SE(theta0,dlog_p,algorithm{i}.opts);
			case 'WNAG'
				[~,out] = WNAG_m(theta0,dlog_p,algorithm{i}.opts);
			case 'MCMC'
				[~,out] = WGF_m(thetaMC,dlog_p,algorithm{i}.opts);
		end
		results{i,j}.name = algorithm{i}.name;
		results{i,j}.Pname = algorithm{i}.Pname;
		results{i,j}.trace = out.trace;
	end
end

save(strcat('./result/',dataset_name,'_MED_id_',num2str(id_test),'.mat'),'results');

