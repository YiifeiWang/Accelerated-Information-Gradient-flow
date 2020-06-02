%%%---------------------------------------------%%%
% This implements the toy example in the numerical experiment.
%%%---------------------------------------------%%%
addpath('./utils');
addpath('./solvers');

p = @(X ) exp(-2*(sqrt(sum(X.^2,2))-3).^2)...
    .*(exp(-2*(X(:,1)-3).^2)+exp(-2*(X(:,1)+3).^2));

xnum = 100; ynum = 100;
xmin = -4; ymin = -4;
xmax = 4; ymax = 4;
x = ([1:xnum]/xnum-0.5)*(xmax-xmin)+(xmax+xmin)/2;
y = ([1:ynum]/ynum-0.5)*(ymax-ymin)+(ymax+ymin)/2;
[X,Y] = meshgrid(x,y);
Z = zeros(xnum,ynum,2);
Z(:,:,1)=X;
Z(:,:,2)=Y;
Z_aux = reshape(Z,[xnum*ynum,2]);
Z_ans = p(Z_aux);
Z_plot = reshape(Z_ans,[xnum,ynum]);

rng(2);
N = 200;
X_init = randn([N,2]);
dlog_p = @dlog_p_toy;
tau = 1e-1;
iter = 200;

% MCMC
opts1 = struct('tau',tau,'iter_num',iter,'ktype',0,'ibw',-1,'ptype',2);
[Xout1,out1] = WGF_m(X_init, dlog_p, opts1);

clf
figure(1);
contourf(X,Y,Z_plot,7);
colormap('white')
hold;
Xp = Xout1(:,1);
Yp = Xout1(:,2);
hp = scatter(Xp,Yp,40,'filled');
set(gcf,'position',[40,40,640,640]);
alpha(hp,0.9);
title('MCMC','FontSize',64);

print('-depsc','./result/toy_bandwidth/MCMC.eps');
saveas(gcf,'./result/toy_bandwidth/MCMC.png');

% MED
opts2 = struct('tau',tau,'iter_num',iter,'ktype',1,'ibw',-1,'ptype',2);
[Xout2,out2] = WGF_m(X_init, dlog_p, opts2);

clf
figure(2);
contourf(X,Y,Z_plot,7);
colormap('white')
hold;
Xp = Xout2(:,1);
Yp = Xout2(:,2);
hp = scatter(Xp,Yp,40,'filled');
set(gcf,'position',[40,40,640,640]);
alpha(hp,0.9);
title('MED','FontSize',64);

print('-depsc','./result/toy_bandwidth/MED.eps');
saveas(gcf,'./result/toy_bandwidth/MED.png');

% HE
opts3 = struct('tau',tau,'iter_num',iter,'ktype',5,'ibw',-1,'ptype',2,'h_pow',3);
[Xout3,out3] = WGF_m(X_init, dlog_p, opts3);

clf
figure(3);
contourf(X,Y,Z_plot,7);
colormap('white')
hold;
Xp = Xout3(:,1);
Yp = Xout3(:,2);
hp = scatter(Xp,Yp,40,'filled');
set(gcf,'position',[40,40,640,640]);
alpha(hp,0.9);
title('HE','FontSize',64);

print('-depsc','./result/toy_bandwidth/HE.eps');
saveas(gcf,'./result/toy_bandwidth/HE.png');

% BM
opts4 = struct('tau',tau,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
[Xout4,out4] = WGF_m(X_init, dlog_p, opts4);

clf
figure(4);
contourf(X,Y,Z_plot,7);
colormap('white')
hold;
Xp = Xout4(:,1);
Yp = Xout4(:,2);
hp = scatter(Xp,Yp,40,'filled');
set(gcf,'position',[40,40,640,640]);
alpha(hp,0.9);
title('BM','FontSize',64);

print('-depsc','./result/toy_bandwidth/BM.eps');
saveas(gcf,'./result/toy_bandwidth/BM.png');

