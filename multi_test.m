%% Relatively comprehensive test
clear 
k1 = 2;
k2 = 3;
a = matexp(eye(k1));
bs = symreal('b',[k1,k1]);
qs = symreal('q',[k1,k2]);
b = matexp('b',bs);
q = matexp('q',qs);


amf = {
        @(a,b,q) b', % OK
        @(a,b,q) a*(2+b*q), %WRONG
        @(a,b,q) a*(2+b'*q),% WRONG
        @(a,b,q) trace(a*(2+b)), %OK
        @(a,b,q) b.^3, %OK
        @(a,b,q) b^2, % OK
        @(a,b,q) trace(b.^3), % OK but adjoint(q) should be ZERO because untouched
        @(a,b,q) inv(b), % wrong
        @(a,b,q) a*(2+inv(b)*q), % wrong        
        @(a,b,q) trace(b.^3)+q, %OK
        @(a,b,q) ones(k1,k2)*trace(b.^3)+q, % OK crashes because we have a scalar solution 
    };
mf = {@(a,b,q) trace(b.^3)*ones(k1,k2)+q};
mf = {@(a,b,q) a.*b};
mf = {@(a,b,q) trace(b.^3)+q};
mf = {@(a,b,q) ones(k1,k2)*trace(b.^3)+q};
mf ={        @(a,b,q) a*(2+b*q)};
mf = {@(a,b,q) inv(b)};
mf = amf;
mf = {@(a,b,q) log(det(b))};
success = zeros(length(mf),2);
for I=1:length(mf)
    disp(mf{I})
    fm = mf{I}(a,b,q);
    fs = mf{I}(value(a),value(b),value(q));
    J_bs = jacobian(fs(:),bs(:));
    J_qs = jacobian(fs(:),qs(:));
    update(fm);
    simplify(value(fm)==fs)

    %   vars = collectvars(fm)
    resetadjoint(b,0); % not needed usually but yes for this test
    resetadjoint(q,0); % not needed usually but yes for this test
    autodiff(fm);
    J_b = adjoint(b);
    J_q = adjoint(q);
    
    % TODO: if J_b is 1 then means IDENTITY not ONES

    wb = simplify(J_b-J_bs)
    wq = simplify(J_q-J_qs)
    try        
        success(I,1) = all(all(double(wb) == 0));
    catch
        warning('Jacobian b is wrong');
    end
    try
        success(I,2) = all(all(double(wq) == 0));
    catch
        warning('Jacobian q is wrong');
    end
    J_b
    J_bs
    J_q
    J_qs
end
success