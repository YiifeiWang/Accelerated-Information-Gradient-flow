%%%---------------------------------------------%%%
% This implements the Gaussian example in the ODE level.
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

% Gaussian example flow 1
A = A1;
tau = 1/max(eig(A))/4;
fprintf('Condition number of W: %.4f\n',cond(A));
sqbeta = sqrt(min(eig(A)));

% WGF
opts1 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0);
[X1,out1] = G_WGF(A,opts1);

% AIG
opts2 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',0,'strong',0);
[X2,out2] = G_AIG(A,opts2);

% AIG-r
opts3 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',1,'strong',0);
[X3,out3] = G_AIG(A,opts3);

% AIG-s
opts4 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',0,'strong',1,'sqbeta',sqbeta);
[X4,out4] = G_AIG(A,opts4);

% AIG-rs
opts5 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',1,'strong',1,'sqbeta',sqbeta);
[X5,out5] = G_AIG(A,opts5);

intervals = ones([10,1])*100;
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
% AIG-s
semilogy_marker(out4.trace.iter,out4.trace.H,markers{4},intervals(4),10,colors{4});
% AIG-rs
semilogy_marker(out5.trace.iter,out5.trace.H,markers{5},intervals(5),10,colors{5});
ylim([1e-8,1e2]);
legend({'WGF','AIG','AIG-r','AIG-s','AIG-rs'},'location','southwest');
xlabel('Iteration');
ylabel('KL divergence');

set(gcf,'position',[40,40,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','./result/Gauss/Gauss_flow1.eps');
saveas(gcf,'./result/Gauss/Gauss_flow1.png');

% Gaussian example flow 2
A = A2;
tau = 1/max(eig(A))/4;
fprintf('Condition number of W: %.4f\n',cond(A));
sqbeta = sqrt(min(eig(A)));

% WGF
opts1 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0);
[X1,out1] = G_WGF(A,opts1);

% AIG
opts2 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',0,'strong',0);
[X2,out2] = G_AIG(A,opts2);

% AIG-r
opts3 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',1,'strong',0);
[X3,out3] = G_AIG(A,opts3);

% AIG-s
opts4 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',0,'strong',1,'sqbeta',sqbeta);
[X4,out4] = G_AIG(A,opts4);

% AIG-rs
opts5 = struct('iter_num',1e3,'tau',tau,'interval',1,'trace',1,'record',0,'crit',0,'restart',1,'strong',1,'sqbeta',sqbeta);
[X5,out5] = G_AIG(A,opts5);

figure(2)
clf;
% WGF
semilogy_marker(out1.trace.iter,out1.trace.H,markers{1},intervals(1),10,colors{1});
% AIG
semilogy_marker(out2.trace.iter,out2.trace.H,markers{2},intervals(2),10,colors{2});
% AIG-r
semilogy_marker(out3.trace.iter,out3.trace.H,markers{3},intervals(3),10,colors{3});
% AIG-s
semilogy_marker(out4.trace.iter,out4.trace.H,markers{4},intervals(4),10,colors{4});
% AIG-rs
semilogy_marker(out5.trace.iter,out5.trace.H,markers{5},intervals(5),10,colors{5});
ylim([1e-10,1e5]);
legend({'WGF','AIG','AIG-r','AIG-s','AIG-rs'},'location','southwest');
xlabel('Iteration');
ylabel('KL divergence');

set(gcf,'position',[40,40,720,360]);
set(gca,'FontSize',16);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
print('-depsc','./result/Gauss/Gauss_flow2.eps');
saveas(gcf,'./result/Gauss/Gauss_flow2.png');
