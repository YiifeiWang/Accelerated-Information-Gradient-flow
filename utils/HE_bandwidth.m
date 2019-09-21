function [ibw_out] = HE_bandwidth(X,ibw_in, HEopts)
%%%---------------------------------------------%%%
% This solves the subproblem in the HE method.
% This code is based on the tensorflow version of HE method
% https://github.com/chang-ml-thu/AWGF
% Understanding and Accelerating Particle-Based Variational Inference
% Chang Liu, Jingwei Zhuo, Pengyu Cheng, Ruiyi Zhang, Jun Zhu, and Lawrence Carin
%%%---------------------------------------------%%%

	[N,d] = size(X);
	if nargin<3; HEopts = []; end
	if ~isfield(HEopts, 'rtype'); HEopts.rtype = 1; end
	if ~isfield(HEopts, 'h_pow'); HEopts.h_pow = d+2; end

	Dij = reshape(X,[N,1,d])-reshape(X,[1,N,d]);
	Dij2 = sum(Dij.^2,3);

	function [gh] = get_obj_ibandw(ibw)
		h = 1/ibw;
		Eij = exp(-Dij2/(2*h)); % N*N*1
		DE = (sum(Dij.*Eij,1)./sum(Eij,1)); %1*N*d;
		DED = sum(DE.*Dij,3); %N*N*1;
		gkh = sum((Dij2-d*h+DED).*Eij*h^(-d/2-2),2);%*h^(-d/2-2)
		gkh = reshape(gkh,[N,1]);
		switch HEopts.rtype
			case 1
				gh = sum(gkh.^2)*h^HEopts.h_pow;
		end
	end

	explore_ratio = 1.1;
	obj_ibw_in = get_obj_ibandw(ibw_in);
	epsi = 1e-6;
	grad_ibw_in = (get_obj_ibandw(ibw_in+epsi)-obj_ibw_in)/epsi;
	if grad_ibw_in<0
		ibw_1 = ibw_in*explore_ratio;
	else
		ibw_1 = ibw_in/explore_ratio;
	end
	obj_ibw_1 = get_obj_ibandw(ibw_1);
	slope_ibw = (obj_ibw_1-obj_ibw_in)/(ibw_1-ibw_in);
	ibw_2 = (ibw_in * slope_ibw - 0.5 * grad_ibw_in * (ibw_1 + ibw_in)) / (slope_ibw - grad_ibw_in);
	obj_ibw_2 = get_obj_ibandw(ibw_2);
	if ~isnan(ibw_2)&&ibw_2>0
		if obj_ibw_1<obj_ibw_in
			if obj_ibw_2<obj_ibw_1
				ibw_out = ibw_2;
			else
				ibw_out = ibw_1;
			end
		else
			if obj_ibw_2<obj_ibw_in
				ibw_out = ibw_2;
			else
				ibw_out = ibw_in;
			end
		end
	else
		if obj_ibw_1<obj_ibw_in
			ibw_out = ibw_1;
		else
			ibw_out = ibw_in;
		end
	end
	

end