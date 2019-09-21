function draw_figures_results_f(results,name,res_id)

% read data
len = length(results);
temp_results        = cell(len,1);

for i = 1: len
temp_results{i}     = results{i}; 
end

res_name            = name;

legend_loc_t = 'southeast';

switch res_name
        
    case 'covtype'
        x_max_tacc   = 6000;    x_max_erel   = 150;   
          
end

intervals = ones([10,1])*6;

markers     = {'.-','d-','*-','>-','s-','<-','^-','v-','>-','o-','*-','.-','s-','d-','^-','v-','>-','<-','p-','h-'};

names = cell(len,1);

for i = 1:len
    names{i}    = temp_results{i}.name;
end


colors = {[0,0,0],[1,0,1],[0,0,1],[0,1,0],[0.9,0.7,0.0],[255,140,0]/255,...     
          [255,71,71]/255,... 
          [0,101,189]/255,...          
          [17,140,17]/255,...       
          [0.9,0.7,0.0], ...        
          [0,101,189]/255,...      
          [1,1,0],[1,1,1],[1,0,1],[0,1,1],[0,1,0],[0,0,1],[0.0,0.3,0.8],[1,0,0]/255};  

%--------------------------------------------------------------------------
% plot: iter // test_acc
%--------------------------------------------------------------------------

figure(1).MenuBar = 'none';
clf

for i = 1:len
    X = temp_results{i}.iter;
    Y = temp_results{i}.test_acc;
    e = temp_results{i}.test_acc_var; 
    plot_marker_err(X,Y,e,markers{i},intervals(i),15,colors{i});
end

legend(names,'location',legend_loc_t);
xlim([0 x_max_tacc])
ylim([0.6,0.78]);
yticks(([0:6]*0.03+0.6));

xlabel('Iteration');
ylabel('Test accuracy')
set(gca,'FontSize',24);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
set(gcf,'position',[40,40,1080,540]);

print('-depsc',strcat('./result/',res_name,'/results_',res_name,'_test_acc_',res_id,'.eps'));
saveas(gcf,strcat('./result/',res_name,'/results_',res_name,'_test_acc_',res_id,'.png'));
%--------------------------------------------------------------------------
% plot: iter // llh
%--------------------------------------------------------------------------

figure(2).MenuBar = 'none';
clf

for i = 1:len
    X = temp_results{i}.iter;
    Y = temp_results{i}.test_llh;
    e = temp_results{i}.test_llh_var; 
    plot_marker_err(X,Y,e,markers{i},intervals(i),15,colors{i});
end

legend(names,'location',legend_loc_t);
xlim([0 x_max_tacc])
ylim([-0.64,-0.5]);

xlabel('Iteration');
ylabel('Log likelihood')
set(gca,'FontSize',24);
grid on;
set(gca,'YMinorGrid','off','YMinorTick','off');
set(gcf,'position',[40,40,1080,540]);

print('-depsc',strcat('./result/',res_name,'/results_',res_name,'_test_llh_',res_id,'.eps'));
saveas(gcf,strcat('./result/',res_name,'/results_',res_name,'_test_llh_',res_id,'.png'));

