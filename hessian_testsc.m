
%%
% Same as in paper:
clear 
k1 = 1;
k2 = 1;
k3 = 1;
k4 = 1;
as = symreal('a',k1);
bs = symreal('b',k2);
cs = symreal('c',k3);
a = matexp('a',as);%rand(size(as)));
b = matexp('b',bs);%rand(size(bs))); 
c = matexp('c',cs);%rand(size(cs)));

mf = {@(a,b,c) (a+exp(b))*(3*b+c*c)}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]

fm = mf{1}(a,b,c);
fs = mf{1}(as,bs,cs);
allvars = [as,bs,cs];
update(fm)
% we flatten
[r,cf] = flatten(fm)
autodiff(fm)
df_a = adjoint(a);
df_b = adjoint(b); % k2*k4  k2*k4 becomes *2 due to adjoint(q)
df_c = adjoint(c);
H = matexp.hessianpush(r,cf)
df_aH = adjoint(a); % k2*k4  k2*k4 becomes *2 due to adjoint(q)
df_bH = adjoint(b);
df_cH = adjoint(c); % k2*k4  k2*k4 becomes *2 due to adjoint(q)

Js = jacobian(fs(:),allvars(:)); % == stacking columnwise the df_bH ... df_qH
Hs = jacobian(Js(:),allvars(:));

