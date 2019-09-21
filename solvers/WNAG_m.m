function [Xs,out] = WNAG(X_init, dlog_p, opts)
%%%---------------------------------------------%%%
% This implements WNAG and Wnes in 
% Accelerated First-order Methods on the Wasserstein Space for Bayesian Inference
% by Liu, Chang and Zhuo, Jingwei and Cheng, Pengyu 
% and Zhang, Ruiyi and Zhu, Jun and Carin, Lawrence
% 
% Input:
% 		X_init --- initial particle positions, N*d matrix
%		dlog_p --- function handle to compute the derivative of log p(x)
%		opts   --- options structure with fields
%				   tau --- step size
%		      iter_num --- number of iterations
%				 ktype --- the type of the kernel
%        		   ibw --- inverse of the bandwidth of the kernel
%                trace --- whether to trace the information during the process
%						   0: no || 1: yes
%				record --- whether to print the recorded inforamtion during the process
%						   0: no || 1: yes
%			   itPrint --- the interval of printing
%			 	 ptype --- the type of the problem
%						   0: Gaussian
%							  opts.A shall NOT be empty
%						   1: Bayesian logistic regression
%							  opts.X_test, opts.y_test shall NOT be empty
%				 rtype --- option in the subproblem of the HE method
%				 ttype --- option in decaying tau
%
%				 wtype --- 1: WNAG 2: Wnes  
%				 alpha --- acceleration parameter for WNAG
%			   acc_bnd --- parameter for Wnes
%			   acc_shk --- parameter for Wnes
%%%---------------------------------------------%%%
	tic;
	[N,d] = size(X_init);
	if nargin < 3; opts = []; end
	if ~isfield(opts, 'tau'); opts.tau = 0.1;  end
	if ~isfield(opts, 'iter_num'); opts.iter_num = 1e3; end
	if ~isfield(opts, 'ktype');  opts.ktype = 1; end
	if ~isfield(opts, 'ibw');  opts.ibw = -1; end

	if ~isfield(opts, 'h_pow'); opts.h_pow = d/2+2; end

	if ~isfield(opts, 'trace'); opts.trace = 0; end
	if ~isfield(opts, 'record'); opts.record = 0; end
	if ~isfield(opts, 'itPrint'); opts.itPrint = 1; end
	if ~isfield(opts, 'ptype'); opts.ptype = 1; end 
	if ~isfield(opts, 'rtype'); opts.rtype = 1; end 

	if ~isfield(opts, 'ttype'); opts.ttype = 1; end
	if ~isfield(opts, 'tau_dec'); opts.tau_dec = 1; end
	if ~isfield(opts, 'tau_itv'); opts.tau_itv = 100; end

	if ~isfield(opts, 'wtype'); opts.wtype = 1; end 
	if ~isfield(opts, 'alpha'); opts.alpha = 3.9; end
	if ~isfield(opts, 'acc_bnd'); opts.acc_bnd = 1e3; end
	if ~isfield(opts, 'acc_shk'); opts.acc_shk = 0.2; end

	tau = opts.tau;
	iter_num = opts.iter_num;
	ktype = opts.ktype;
	ibw = opts.ibw;


	record = opts.record;
	itPrint = opts.itPrint;
	ptype = opts.ptype;

	X = X_init;

	if opts.trace
		switch ptype
			case 0
				trace_ = zeros(floor(iter_num/itPrint)+1,1);
			case 1
				trace_.test_acc = zeros(floor(iter_num/itPrint)+1,1);
				trace_.test_llh = zeros(floor(iter_num/itPrint)+1,1);
				trace_.iter = zeros(floor(iter_num/itPrint)+1,1);
				trace_.time = zeros(floor(iter_num/itPrint)+1,1);
		end
	end

	if opts.ptype == 0
		A = opts.A;
		logdetA = log(det(A));
	end

	if record == 1
	    % set up print format
	    if ispc; str1 = '  %10s'; str2 = '  %6s';
	    else     str1 = '  %10s'; str2 = '  %6s'; end
	    stra = ['%5s',str2,str2,str2,str1, str2,'\n'];
	    str_head = sprintf(stra, ...
	        'iter', 'TBA','TBA','TBA', 'TBA', 'TBA');
	    str_num = ['%4d | %+2.4e %+2.4e %+2.4e %2.1e %2.1e \n'];
	end
	


	Y = X;
	xi = 0;
	k = 1;
	eva_time = 0;
	for iter = 1:iter_num
		switch opts.ttype
			case 1
				tau = opts.tau*opts.tau_dec^(floor(iter/opts.tau_itv));
			case 2
				tau = opts.tau/iter^opts.tau_dec;
		end
		grad = dlog_p(Y);
		kopts = struct('iter',iter,'rtype',opts.rtype,'ibw',ibw,'h_pow',opts.h_pow,'tau',tau);
		[xi, ibw_out] = dlog_p_prac(Y,ktype,kopts);
		ibw = ibw_out;
		Xp = X; Yp = Y;
		X = Y+tau*(grad-xi);
		switch opts.wtype
			case 1
				Y = X+(k-1)/k*(Yp-Xp)+(k+opts.alpha-2)/k*tau*(grad-xi);
				k = k+1;
			case 2
				muH = opts.acc_bnd*tau;
				beta_ = opts.acc_shk*sqrt(muH);
				Y = X+(1+beta_-2*(1+beta_)*(2+beta_)*muH/...
				(sqrt(beta_^2+4*(1+beta_)*muH)-beta_+2*(1+beta_)*muH))*(X-Xp);
		end
		if opts.trace && (iter == 1 || mod(iter,itPrint)==0)
			switch ptype
				case 0
					Sigma = cov(X);
					if cond(Sigma)<1e-12;
						break
					end
					tmp = A*Sigma;
					H = (-d-log(det(Sigma))-logdetA+trace(tmp))/2;
					trace_(floor(iter/itPrint)+1) = H;
					trace_.iter(floor(iter/itPrint)+1) = iter;
				case 1
					tmp1 = toc;
					[test_acc, test_llh] = bayeslr_evaluation(X, opts.X_test, opts.y_test);
					eva_time = eva_time+toc-tmp1;
					trace_.test_acc(floor(iter/itPrint)+1) = test_acc;
					trace_.test_llh(floor(iter/itPrint)+1) = test_llh;
					trace_.iter(floor(iter/itPrint)+1) = iter;
					trace_.time(floor(iter/itPrint)+1) = toc-eva_time;
				case 2
					trace_(floor(iter/itPrint)+1) = ibw;
			end
			if record
		        if iter == 1 || mod(iter,20*itPrint) == 0 
		            fprintf('\n%s', str_head);
		        end
		        if iter == 1 || mod(iter,itPrint) == 0
		        	switch ptype
						case 0
							fprintf(str_num,iter,H, min(eig(Sigma)), max(eig(Sigma)),ibw,0);
						case 1
							fprintf(str_num,iter,test_acc, test_llh, k,0,0);
						case 2
							fprintf(str_num,iter,ibw, 0, 0,0,0);
					end
		        end
			end
		end

	Xs = X;
	out = [];
	if opts.trace;
		out.trace = trace_;
	end
end