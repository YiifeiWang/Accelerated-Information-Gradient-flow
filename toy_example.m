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
X_aux = randn([N,1]);
X_init = randn([N,2])+[0,10];
dlog_p = @dlog_p_toy;

save_path = './result/toy/';

%iter = 20;
%iter = 200;
iters = [20,40,80,160];
wid = 320;
hei = 320;
for i = 1:length(iters)
	iter = iters(i);

	opts1 = struct('tau',0.1,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2);
	[Xout1,out1] = WGF_m(X_init, dlog_p, opts1);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout1(:,1);
	Yp = Xout1(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	hold off;
	title(strcat('W-GF Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'WGF_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'WGF_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);

	opts2 = struct('tau',0.1,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'strong',0,'restart',1);
	[Xout2,out2] = W_AIG(X_init, dlog_p, opts2);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout2(:,1);
	Yp = Xout2(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	hold off;
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	title(strcat('W-AIG Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'WAIG_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'WAIG_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);

	opts1 = struct('tau',0.02,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'lbd',1);
	[Xout1,out1] = KWGF_m(X_init, dlog_p, opts1);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout1(:,1);
	Yp = Xout1(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	hold off;
	title(strcat('KW-GF Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'KWGF_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'KWGF_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);

	opts2 = struct('tau',0.02,'iter_num',iter,'ktype',6,'ibw',-1,'ptype',2,'strong',0,'restart',1,'lbd',1);
	[Xout2,out2] = KW_AIG(X_init, dlog_p, opts2);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout2(:,1);
	Yp = Xout2(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	hold off;
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	title(strcat('KW-AIG Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'KWAIG_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'KWAIG_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);

	opts1 = struct('tau',0.1,'iter_num',iter,'ktype',3,'ibw',1,'ptype',2,'adagrad',1);
	[Xout1,out1] = SVGD_m(X_init, dlog_p, opts1);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout1(:,1);
	Yp = Xout1(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	hold off;
	title(strcat('SVGD Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'SVGD_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'SVGD_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);

	opts2 = struct('tau',0.1,'iter_num',iter,'ktype',3,'ktype_inner',6,'ibw',1,'ptype',2,'strong',0,'restart',1);
	[Xout2,out2] = S_AIG(X_init, dlog_p, opts2);

	clf
	figure(1);
	contourf(X,Y,Z_plot,7);
	colormap('white')
	hold on;
	Xp = Xout2(:,1);
	Yp = Xout2(:,2);
	hp = scatter(Xp,Yp,40,'filled');
	hold off;
	set(gcf,'position',[0,0,wid,hei]);
	alpha(hp,0.9);
	title(strcat('S-AIG Iter: ',mat2str(iter)),'FontSize',36);
	save_path1 = strcat(save_path,'SAIG_iter',mat2str(iter),'.eps');
	print('-depsc',save_path1);
	save_path2 = strcat(save_path,'SAIG_iter',mat2str(iter),'.png');
	saveas(gcf,save_path2);
end



