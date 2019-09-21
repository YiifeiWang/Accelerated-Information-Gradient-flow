%%%---------------------------------------------%%%
% This implements the Gaussian example in the particle level.
%%%---------------------------------------------%%%
addpath('./utils');
addpath('./solvers');

rng(2)
N = 600;
d = 100;
aux = randn([d,d]);
A1 = aux'*aux+1e-1*eye(d);
A1 = A1/max(eig(A1));

aux = randn([d,d]);
A2 = aux'*aux+1e-1*eye(d);
A2 = A2/min(eig(A2));

% Gaussian example 1
X_init = randn([N,d]);
A = A1;
tau = 1/max(eig(A))/4;
fprintf('Condition number of W: %.4f\n',cond(A));
sqbeta = sqrt(min(eig(A)));
dlog_p = @(X) dlog_p_Gaussian(X,A);
iter = 1e3;

% WGF
opts1 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A);
[Xout1,out1] = WGF_m(X_init,dlog_p,opts1);
% AIG
opts2 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',0,'strong',0);
[Xout2,out2] = AIG_SE(X_init,dlog_p,opts2);
% AIG-r
opts3 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',1,'strong',0);
[Xout3,out3] = AIG_SE(X_init,dlog_p,opts3);
% AIG(s)
opts4 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',0,'strong',1,'sqbeta',sqbeta);
[Xout4,out4] = AIG_SE(X_init,dlog_p,opts4);
% AIG(s)-r
opts5 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',1,'strong',1,'sqbeta',sqbeta);
[Xout5,out5] = AIG_SE(X_init,dlog_p,opts5);

intervals = ones([10,1])*10;
markers     = {'d-','*-','s-','<-','^-','*-','v-','>-','o-','*-','.-','s-','d-','^-','v-','>-','<-','p-','h-'};
colors = {[0,0,1],[1,0,1],[0,1,0],...     
          [255,71,71]/255,... 
          [0.9,0.7,0.0],...
          [0,101,189]/255,...          
          [17,140,17]/255,...       
          [0.9,0.7,0.0], ...        
          [0,101,189]/255,...       
          [1,1,0],[1,1,1],[1,0,1],[0,1,1],[0,1,0],[0,0,1],[0.0,0.3,0.8],[1,0,0]/255};  


figure(1)
clf;
% WGF
semilogy_marker(out1.trace.iter,out1.trace.H,markers{1},intervals(1),10,colors{1});
% AIG
semilogy_marker(out2.trace.iter,out2.trace.H,markers{2},intervals(2),10,colors{2});
% AIG-r
semilogy_marker(out3.trace.iter,out3.trace.H,markers{3},intervals(3),10,colors{3});
% AIG(s)
semilogy_marker(out4.trace.iter,out4.trace.H,markers{4},intervals(4),10,colors{4});
% AIG(s)-r
semilogy_marker(out5.trace.iter,out5.trace.H,markers{5},intervals(5),10,colors{5});
ylim([1e-8,1e2]);
legend({'WGF','AIG','AIG-r','AIG-s','AIG-rs'},'location','southwest');
xlabel('Iteration');
ylabel('KL divergence');

set(gcf,'position',[40,40,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','./result/Gauss/Gauss_1.eps');
saveas(gcf,'./result/Gauss/Gauss_1.png');


% Gaussian example 2
X_init = randn([N,d]);
A = A2;
tau = 1/max(eig(A))/4;
fprintf('Condition number of W: %.4f\n',cond(A));
sqbeta = sqrt(min(eig(A)));
dlog_p = @(X) dlog_p_Gaussian(X,A);
iter = 1e3;

opts1 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A);
[Xout1,out1] = WGF_m(X_init,dlog_p,opts1);

opts2 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',0,'strong',0);
[Xout2,out2] = AIG_SE(X_init,dlog_p,opts2);

opts3 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',1,'strong',0);
[Xout3,out3] = AIG_SE(X_init,dlog_p,opts3);

opts4 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',0,'strong',1,'sqbeta',sqbeta);
[Xout4,out4] = AIG_SE(X_init,dlog_p,opts4);

opts5 = struct('tau',tau,'iter_num',1e3,'trace',1,'itPrint',10,'ktype',3,'ptype',0,'A',A,'restart',1,'strong',1,'sqbeta',sqbeta);
[Xout5,out5] = AIG_SE(X_init,dlog_p,opts5);


figure(2)
clf;
% WGF
semilogy_marker(out1.trace.iter,out1.trace.H,markers{1},intervals(1),10,colors{1});
% AIG
semilogy_marker(out2.trace.iter,out2.trace.H,markers{2},intervals(2),10,colors{2});
% AIG-r
semilogy_marker(out3.trace.iter,out3.trace.H,markers{3},intervals(3),10,colors{3});
% AIG(s)
semilogy_marker(out4.trace.iter,out4.trace.H,markers{4},intervals(4),10,colors{4});
% AIG(s)-r
semilogy_marker(out5.trace.iter,out5.trace.H,markers{5},intervals(5),10,colors{5});
ylim([1e-10,1e8]);
legend({'WGF','AIG','AIG-r','AIG-s','AIG-rs'},'location','southwest');
xlabel('Iteration');
ylabel('KL divergence');

set(gcf,'position',[40,40,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','./result/Gauss/Gauss_2.eps');
saveas(gcf,'./result/Gauss/Gauss_2.png');
