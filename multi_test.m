%% Relatively comprehensive test
clear 
k = 2;
a = matexp(eye(k));
bs = symreal('b',[k,k]);
qs = symreal('q',[k,k]);
b = matexp('b',bs);
q = matexp('q',qs);


mf = {
        @(a,b,q) b', % wrong
        @(a,b,q) a*(2+b*q),
        @(a,b,q) a*(2+b'*q),
        @(a,b,q) trace(a*(2+b*q)),
        @(a,b,q) trace(b.^3), % OK but adjoint(q) should be ZERO because untouched
        @(a,b,q) inv(b), % wrong
        %@(a,b,q) a*(2+inv(b)*q), % wrong
        %FAIL MUPAD @(a,b,q) b^2
        %FAIL MUPAD @(a,b,q) b.^3,
        %@(a,b,q) trace(b.^3)+q, % crashes
    };
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

    wb = simplify(J_b-J_bs)
    wq = simplify(J_q-J_qs)
    try        
        success(I,1) = all(all(double(wb)) == 0);
    catch
        warning('Jacobian b is wrong');
    end
    try
        success(I,2) = all(all(double(wq)) == 0);
    catch
        warning('Jacobian q is wrong');
    end
end
success