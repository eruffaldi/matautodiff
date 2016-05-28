
%%
% Scalar function as in paper
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

mf = {@(a,b,c) (a+exp(b))*(3*b+c^2)}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]
mf = {@(a,b,c) (a+exp(b))*(3*b+c*c)}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]


fm = mf{1}(a,b,c);
fs = mf{1}(as,bs,cs);
allvars = [as,bs,cs];
update(fm)
% we flatten
[r,cf] = flatten(fm)
autodiff(fm)

%fs = (a+exp(b))*(3*b+c*c)
df_a = adjoint(a); % c1^2 + 3*b1  OK
df_b = adjoint(b); % 3*a1 + 3*exp(b1) + exp(b1)*(c1^2 + 3*b1) WRONG
    % (a+exp(b)) -- (3*b+c*c)
    %    ramo  (a+exp(b)) viene  (a+exp(b))*(3->b + c->c + c->c)
    %    ramo (3*b+c*c)   viene  (1->a + exp(b)->b)
df_c = adjoint(c); % 2*c1*(a1 + exp(b1)) OK
H = matexp.hessianpush(r,cf);
df_aH = adjoint(a); % OK
df_bH = adjoint(b); % OK: 3*a1 + 3*exp(b1) + exp(b1)*(c1^2 + 3*b1)
df_cH = adjoint(c); % OK with ^2

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