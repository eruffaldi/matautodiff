% Work in progress for hessian test
clear 
k1 = 2;
k2 = 3;
k3 = 4;
k4 = 5;
a = matexp(eye(3));
bs = symreal('b',[k1,k2]);
qs = symreal('q',[k2,k4]);
b = matexp('b',bs);
q = matexp('q',qs);

mf = {@(a,b,q) b*q}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]

f = mf{1}(a,b,q);
update(f)
xautodiff(f)
df_b = adjoint(b)
%autodiff(df_b)
%df_bb = adjoint(b);
%df_bb