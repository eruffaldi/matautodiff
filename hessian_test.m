% Work in progress for hessian test
clear 
k1 = 2;
k2 = 3;
k3 = 4;
k4 = 5;
a = matexp(eye(3));
bs = symreal('b',[k2,k4]);
qs = symreal('q',[k2,k4]);
b = matexp('b',bs);
q = matexp('q',qs);

mf = {@(a,b,q) b+2*q}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]

f = mf{1}(a,b,q);
fs = mf{1}(value(a),value(b),value(q));
allvars = [bs(:);qs(:)];
update(f)
% we flatten
[r,c] = flatten(f)
autodiff(f)
df_b = adjoint(b) % k2*k4  k2*k4 becomes *2 due to adjoint(q)
df_q = adjoint(q)
H = matexp.hessianpush(r,c)

adjoint(b)
adjoint(q)
error('ciao')
%%
J_s = jacobian(fs(:),allvars(:)); % k2*k4*k2*k4 , 2*k2*k4
J_ss = jacobian(J_s(:),allvars(:)); % [kl mn,mn], k2*k4*k2*k4 *  2*k2*k2 , 2*k2*k4

update(f)
autodiff(f,1)
df_b = adjoint(b); % k2*k4  k2*k4 becomes *2 due to adjoint(q)
