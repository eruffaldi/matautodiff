%% b*q compared against symbolic
clear 
k1 = 2;
k2 = 3;
k3 = 4;
k4 = 5;
a = matexp(eye(3));

bs = symreal('b',[k2,k4]);
qs = symreal('q',[k4,k1]);
b = matexp('b',rand(size(bs))); 
q = matexp('q',rand(size(qs)));

mf = {@(a,b,q) b*q}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]

fm = mf{1}(a,b,q);
fs = mf{1}(value(a),bs,qs);
allvars = [bs(:);qs(:)];
update(fm)
% we flatten
[r,c] = flatten(fm)
autodiff(fm)
df_b = adjoint(b) % k2*k4  k2*k4 becomes *2 due to adjoint(q)
df_q = adjoint(q)
H = matexp.hessianpush(r,c)
df_bH = adjoint(b) % k2*k4  k2*k4 becomes *2 due to adjoint(q)
df_qH = adjoint(q)


ya=H{2,1};
yb=H{1,2};
% symmetry check
wq = [];
for I=1:size(ya,1)
    wa = ya(I,:,:);
    wb = squeeze(yb(I,:,:))';
    wq(I) = all(wa(:) == wb(:));
end
wq

J_s = jacobian(fs(:),allvars(:)); % == stacking columnwise the df_bH ... df_qH
H_s = double(jacobian(J_s(:),allvars(:)));
% fs,[b,q],[b,q]
H_sT = reshape(H_s,numel(fm),numel(b)+numel(q),numel(b)+numel(q));

size(H_sT)
H_sT_qb = H_sT(:,1:numel(b),numel(b)+1:end);
H_sT_bq = H_sT(:,numel(b)+1:end,1:numel(b));
size(H{2,1})
size(H_sT_qb)

wq = [];
for I=1:size(H_sT_qb,1)
    wx = squeeze(H_sT_qb(I,:,:));
    wy = squeeze(H{2,1}(I,:,:));
    wq(I) = all(wx(:)==wy(:));
end
wq
