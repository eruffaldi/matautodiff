%% Relatively comprehensive test
clear 
k1 = 2;
k2 = 3;
k3 = 4;
k4 = 5;
a = matexp(eye(3));
bs = symreal('b',[k1,k2]);
qs = symreal('q',[k3,k4]);
b = matexp('b',bs);
q = matexp('q',qs);

mf = {@(a,b,q) kron(b,q)}; % [k1,k1] kr [k1,k2] = [k1k1,k1k2] => [k1^3 k2]
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

%% Tensor version of the kronecker
k1 = 2;
k2 = 3;
k3 = 5;
R = symreal('b',[k2,k1]); % 
q = kron(R',eye(k3));
cc = cell(size(q,1),size(q,2));
zz = zeros(size(q,1),size(q,2),numel(R));
for I=1:size(q,1)
    for J=1:size(q,2)
        zz(I,J,:) = double(jacobian(q(I,J),R(:)));
    end
end
W = double(matexp.dkronRT(R,k3,1));
size(W)==size(zz)
sum(W(:)-zz(:))
W1=W;


%% Matrix version of the kronecker
k1 = 2;
k2 = 3;
k3 = 5;
R = symreal('b',[k2,k1]); % 
q = kron(R',eye(k3));
zz = zeros(size(q,1),size(q,2),numel(R));
for I=1:size(q,1)
    for J=1:size(q,2)
        % equivalent zz(I,sub2ind([size(q,2),numel(L)],repmat(J,1,numel(L)),1:numel(L))) = double(jacobian(q(I,J),L(:)));
        zz(I,J,:) = double(jacobian(q(I,J),R(:)));
    end
end
zz = reshape(zz,size(zz,1),[]);
W = double(matexp.dkronR(R,k3,1));
size(W)==size(zz)
sum(W(:)-zz(:))
W1=W;

%% Tensor version of the kronecker JLL general
k1 = 2;
k2 = 3;
k3 = 5;
L = symreal('a',[k3,k2]); % 
R = symreal('b',[k2,k1]); % 
q = kron(R',eye(k3));
zz = zeros(size(q,1),size(q,2),numel(L));
for I=1:size(q,1)
    for J=1:size(q,2)
        zz(I,J,:) = double(jacobian(q(I,J),L(:)));
    end
end

%% Tensor version of the kronecker
k1 = 2;
k2 = 3;
k3 = 5;
L = symreal('b',[k3,k2]); % k3*k1
q = kron(eye(k1),L);
cc = cell(size(q,1),size(q,2));
zz = zeros(size(q,1),size(q,2),numel(L));
for I=1:size(q,1)
    for J=1:size(q,2)
        zz(I,J,:) = double(jacobian(q(I,J),L(:)));
    end
end
zz2 = reshape(zz,size(q,1),[]);
W = double(matexp.dkronLT(L,k1,1));
size(W)==size(zz)
sum(zz(:)-W(:))
W2=W;


%% Matrix version of the kronecker
k1 = 2;
k2 = 3;
k3 = 5;
L = symreal('b',[k3,k2]); % k3*k1
q = kron(eye(k1),L);
zz = zeros(size(q,1),size(q,2)*numel(L));
for I=1:size(q,1)
    for J=1:size(q,2)
        zz(I,sub2ind([size(q,2),numel(L)],repmat(J,1,numel(L)),1:numel(L))) = double(jacobian(q(I,J),L(:)));
    end
end
%zz = reshape(zz,size(zz,1),[]);

W = double(matexp.dkronL(L,k1,1));
size(W)==size(zz)
sum(zz(:)-W(:))
W2=W;