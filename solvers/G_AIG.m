function [X,out] = G_AIG(A,opts)
% implement the AIG flow in Gaussian
if nargin<2; opts = []; end
[n,~] = size(A);
I = eye(n);
if ~isfield(opts, 'tau'); opts.tau = 0.1; end
if ~isfield(opts, 'X');   opts.X   = I;   end
if ~isfield(opts, 'iter_num');   opts.iter_num   = 100;        end
if ~isfield(opts, 'record');     opts.record     = 0;          end
if ~isfield(opts, 'itPrint');    opts.itPrint    = 10;         end
if ~isfield(opts, 'Htol');       opts.Htol       = 1e-16;       end
if ~isfield(opts, 'trace');      opts.trace     = 0;          end
if ~isfield(opts, 'interval');   opts.interval   = 10;        end
if ~isfield(opts, 'crit');       opts.crit       = 0;           end
if ~isfield(opts, 'restart');  opts.restart  = 0;           end
if ~isfield(opts, 'sqbeta');     opts.sqbeta      = 1;          end
if ~isfield(opts, 'strong');     opts.strong        = 0;        end

tau 	 = opts.tau;
iter_num = opts.iter_num;
X 		 = opts.X;
Htol     = opts.Htol;
record   = opts.record;
itPrint  = opts.itPrint;
trace_   = opts.trace;
interval = opts.interval;

sqbeta   = opts.sqbeta;
strong   = opts.strong;

trace_data = [];
trace_data.iter = zeros(floor(iter_num/interval),1);
trace_data.H = zeros(floor(iter_num/interval),1);

if record == 1
    % set up print format
    if ispc; str1 = '  %10s'; str2 = '  %6s';
    else     str1 = '  %10s'; str2 = '  %6s'; end
    stra = ['%5s',str2,str2,str2,str1, str2,'\n'];
    str_head = sprintf(stra, ...
        'iter', 'H','F_norm','inf_norm', 'Ymin', 'Xmin');
    str_num = ['%4d | %+2.4e %+2.4e %+2.4e %2.1e %2.1e \n'];
end

S = 0;
k = 0;
cstop = 0;
I = eye(n);
sqtau = sqrt(tau);
coef = (1/sqtau-sqbeta)/(sqbeta+1/sqtau);
for iter=1:iter_num
    grad = -inv(X)+A;
    if strong
        S = coef*S-sqtau*(grad+S*S);
    else
        S = (k-1)/(k+2)*S-sqtau*(grad+S*S);
    end
    if -trace(S*X*grad)>=0 || ~opts.restart
        k = k+1;
    else
        S = -sqtau*grad;
        k = 1;
    end
    if trace(tau*S*S)>n %&&opts.adapt_rst
        S = -sqtau*grad;
    end
    aux = I+sqtau*S;
    X = aux*X*aux;
    tmp = A*X;
    H = -(n/2 + 0.5*log(det(tmp))-0.5*trace(tmp));
    switch opts.crit
        case 1
            cstop = (H<Htol);
    end
	if (record == 1) && (iter == 1 || mod(iter,itPrint)==0 || cstop)
        if iter == 1 || mod(iter,20*itPrint) == 0 
            fprintf('\n%s', str_head);
        end
        if iter == 1 || mod(iter,itPrint) == 0
            fprintf(str_num,iter,H, 0,0,0,0);
        end
    end
    if trace_ && mod(iter,interval) == 0
        trace_data.H(floor(iter/interval)) = H;
        trace_data.iter(floor(iter/interval)) = iter;
    end
    if cstop
        break
    end

end

out.trace.H = trace_data.H(1:floor(iter/interval));
out.trace.iter = trace_data.iter(1:floor(iter/interval));