% Sinquad function CUTE as used in ref paper
%   f(x) = (x1-1)^4 + sum_(i=2..n-1) ( sin (xi-xn) - x1^2 + xi^2)^2 +
%   (xn^2-x1^2)^2
%   x0 = all 0.1
%   
% Matrix form: (x1,xn) scalar, (x2..xn-1)=X vector
%   A) f(x1,xn,X) = ... + sum (sin(X-xn) - x1^2 + X.^2).^2 + ...
%       with sum function (missing)
%   B) f(x1,xn,X) = ... + Y*Y' + ...
%           Y = (sin(X-xn) - x1^2 + X.^2)  with subexpression explicit
%   C) f(x1,xn,X) = ... + (....)*(...)'+...
%           not explicit subexpression

%%
% Same as in paper:
clear 
k1 = 1;
k2 = 1;
k3 = 2;
k4 = 1;
x1s = symreal('x1',k1);
xns = symreal('xn',k2);
xis = symreal('xi',k3);
x1 = matexp('x1',x1s);%rand(size(as)));
xn = matexp('xn',xns);%rand(size(bs))); 
xi = matexp('xi',xis);%rand(size(cs)));

mf = {@(x1,xi,xn) (x1-1)^4+ (sin(xi-xn)-x1^2+xi.^2)*(sin(xi-xn)-x1^2+xi.^2)' + (xn^2-x1^2)^2 };

fm = mf{1}(x1,xi,xn);
fs = mf{1}(x1,xi,xn);
update(fm)
% we flatten
[r,cf] = flatten(fm)
autodiff(fm)

df_x1 = adjoint(x1); 
df_xn = adjoint(xn); 
df_xi = adjoint(xi); 
H = matexp.hessianpush(r,cf);
df_x1H = adjoint(x1); % OK
df_xnH = adjoint(xn); % OK: 3*a1 + 3*exp(b1) + exp(b1)*(c1^2 + 3*b1)
df_xiH = adjoint(xi); % OK with ^2

allvars = zeros(cf(2),1,'like',as);
for I=1:cf(2)
    allvars(I) = value(r{cf(1)+I,1});
end

Js = jacobian(fs(:),allvars(:)); % == stacking columnwise the df_bH ... df_qH
Hs = jacobian(Js(:),allvars(:));
HH = zeros(size(H),'like',as);
% symmetrize
for I=1:size(H,1)
    for J=I:size(H,2)
        HH(I,J) = H{J,I};
        HH(J,I) = HH(I,J);
    end
end
% reorder allvars vs 
HH
Hs