function [res] = MMD(X,Y,ktype,kopts)
%%%---------------------------------------------%%%
% This computes the MMD between particles systems X and Y
%%%---------------------------------------------%%%
	if nargin<4; kopts = [];  end
	if ~isfield(kopts, 'h');  kopts.h = 1;  end

	[N,d] = size(X);
	[M,d] = size(Y);

	switch ktype
		case 1
			h = kopts.h;
			Xtmp1 = reshape(X, [N,1,d]);
			Xtmp2 = reshape(X, [1,N,d]);
			Ytmp1 = reshape(Y, [M,1,d]);
			Ytmp2 = reshape(Y, [1,M,d]);
			Dxx = Xtmp1-Xtmp2;
			Dxy = Xtmp1-Ytmp2;
			Dyy = Ytmp1-Ytmp2;
			Kxx = exp(-sum(Dxx.^2,3)/(2*h));
			Kxy = exp(-sum(Dxy.^2,3)/(2*h));
			Kyy = exp(-sum(Dyy.^2,3)/(2*h));
			res = 1/N^2*sum(Kxx(:))-2/(M*N)*sum(Kxy(:))+1/M^2*sum(Kyy(:));
			res = res*(2*pi*h)^(-d/2);
	end