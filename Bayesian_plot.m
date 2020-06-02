%%%---------------------------------------------%%%
% This plots the results in Bayesian logistic regression.
%%%---------------------------------------------%%%
addpath('./utils');

load('./result/covtype_BM_id_1.mat');
result_a = get_plt(results);
time_a = get_time(results);
draw_figures(result_a,'covtype','BM');
print_time(time_a, result_a,'BM');

load('./result/covtype_MED_id_1.mat');
result_a = get_plt(results);
time_a = get_time(results);
draw_figures(result_a,'covtype','MED');
print_time(time_a, result_a,'MED');


function [result_a] = get_plt(results)
    [N_alg, N_test] = size(results);
    result_a = {};
    iter = [1,[1:60]*100]';
    for i = 1:N_alg
        result_a{i}.test_acc = 0;
        result_a{i}.test_acc_var = 0;
        result_a{i}.test_llh = 0;
        result_a{i}.test_llh_var = 0;
        result_a{i}.name = results{i,1}.Pname;
        result_a{i}.iter = iter;
        result_a{i}.num = 0;
    end
    for i = 1:N_alg
        for j = 1:N_test
            if sum(results{i,j}.trace.test_acc(end-10:end))~=0 && ~isnan(sum(results{i,j}.trace.test_llh(end-10:end)))
                result_a{i}.test_acc = result_a{i}.test_acc+results{i,j}.trace.test_acc;
                result_a{i}.test_llh = result_a{i}.test_llh+results{i,j}.trace.test_llh;
                result_a{i}.num = result_a{i}.num+1;
            end
        end
        result_a{i}.test_acc = result_a{i}.test_acc/result_a{i}.num;
        result_a{i}.test_llh = result_a{i}.test_llh/result_a{i}.num;
        for j = 1:N_test
            if sum(results{i,j}.trace.test_acc(end-10:end))~=0 && ~isnan(sum(results{i,j}.trace.test_llh(end-10:end)))
                result_a{i}.test_acc_var = result_a{i}.test_acc_var+...
                    (result_a{i}.test_acc-results{i,j}.trace.test_acc).^2;
                result_a{i}.test_llh_var = result_a{i}.test_llh_var+...
                    (result_a{i}.test_llh-results{i,j}.trace.test_llh).^2;
            end
        end
        result_a{i}.test_acc_var = sqrt(result_a{i}.test_acc_var/(result_a{i}.num-1));
        result_a{i}.test_llh_var = sqrt(result_a{i}.test_llh_var/(result_a{i}.num-1));
    end
end

function [time_a] = get_time(results)
    [N_alg, N_test] = size(results);
    time_a = zeros(N_alg,1);
    for i = 1:N_alg
        for j = 1:N_test
            time_a(i) = time_a(i)+results{i,j}.trace.time(61);
        end
        time_a(i) = time_a(i)/N_test;
    end
end

function [] = print_time(time_a,results_a,method)
    [N_alg, ~] = size(time_a);
    fprintf('%s\n',method);
    for i = 1:N_alg
        fprintf('%8s ',results_a{i}.name);
    end
    fprintf('\n');
    for i = 1:N_alg
        fprintf('%8.3f ',time_a(i));
    end
    fprintf('\n');
end