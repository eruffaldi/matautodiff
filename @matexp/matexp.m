% Matrix Expression Class for Automatic Differentiation
% Emanuele Ruffaldi 2016, Scuola Superiore Sant'Anna
%
% Provides Reverse-Mode Automatic Differentiation of Matrices
% with support for Hessian Matrix computation based on edge-push algorithm
%
classdef matexp < handle
    
    properties
        aname       % name of the variable
        avalue      % value of constant or in general of the term
        aadjoint    % adjoint for intermediate, and derivative for terminals
        aop         % operations string
        aoperands   % operands
        avarcount   % number of depending variables
    end
    
    methods
        % Possible constructors:
        % matexp(value)
        % matexp(name,value)
        % matexp([],op,operands)
        function this = matexp(varargin)
            if nargin == 2
                this.aname = varargin{1};
                this.avalue = varargin{2};
                this.aop= '';
                this.avarcount = numel(this.avalue);
            elseif nargin == 1
                this.avalue = varargin{1};
                this.aop= '';
                this.avarcount = 0;
            elseif nargin == 3
                % skip first
                this.aop = varargin{2};
                this.aoperands = varargin{3};
                this.avarcount = 0;
                for I=1:length(this.aoperands)
                    this.avarcount = this.avarcount + this.aoperands{I}.avarcount;
                end
            else
                error('Invalid matexp constructor');
            end
        end
        
        function autodiff(this)
            if isempty(this.avalue)
                update(this);
            end
            resetadjoint(this,0);
            % [k,m] -> [k,m] => eye [k,m,k,m] => [km,km] jacobian matrix
            this.aadjoint = eye(numel(this.avalue));
            mautodiff(this);
        end
        
        % collects the variables present in the expression tree
        function r = collectvars(this)
            r = ucollectvars(this);
        end
        
        % INTERNAL recursive of collectvars
        function r = ucollectvars(this)
            r = {};
            if  ~isempty(this.aname)
                r = {this};
            else
                for I=1:length(this.aoperands)
                    w = collectvars(this.aoperands{I});
                    r = [r; w];
                end
            end
        end
        
        % resets he adjoint before autodiff
        function resetadjoint(this,x)
            for I=1:length(this.aoperands)
                resetadjoint(this.aoperands{I},x);
            end
            this.aadjoint = x;
        end
        
        % updates the value
        function update(this)
            for I=1:length(this.aoperands)
                update(this.aoperands{I});
            end
            switch(this.aop)
                case 'exp'
                    assert(length(this.aoperands{1}.avalue) == 1);
                    this.avalue = exp(this.aoperands{1}.avalue);
                case '+'
                    this.avalue = this.aoperands{1}.avalue+this.aoperands{2}.avalue;
                case 'u-'
                    this.avalue = -this.aoperands{1}.avalue;
                case '-'
                    this.avalue = this.aoperands{1}.avalue-this.aoperands{2}.avalue;
                case '*'
                    this.avalue = this.aoperands{1}.avalue*this.aoperands{2}.avalue;
                case '.*'
                    this.avalue = this.aoperands{1}.avalue.*this.aoperands{2}.avalue;
                case 'inv'
                    this.avalue = inv(this.aoperands{1}.avalue);
                case 'transpose'
                    this.avalue = (this.aoperands{1}.avalue)';
                case 'trace'
                    this.avalue = trace(this.aoperands{1}.avalue);
                case 'power'
                    this.avalue = this.aoperands{1}.avalue.^this.aoperands{2}.avalue;
                case 'mpower'
                    this.avalue = this.aoperands{1}.avalue^this.aoperands{2}.avalue;
                case 'cos'
                    this.avalue = cos(this.aoperands{1}.avalue);
                case 'sin'
                    this.avalue = sin(this.aoperands{1}.avalue);
                case 'logdet'
                    this.avalue = log(det(this.aoperands{1}.avalue));
                case 'kron'
                    this.avalue = kron(this.aoperands{1}.avalue,this.aoperands{2}.avalue);
                case 'det'
                    this.avalue = det(this.aoperands{1}.avalue);
                case 'diag'
                    this.avalue = diag(this.aoperands{1}.avalue);
                case 'vec'
                    c = this.aoperands{1}.avalue;
                    this.avalue = c(:);
                case '' % leaf
                otherwise
                    error('unsupported operator');
            end
        end
        
        % Overloading a+b
        function r = plus(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'+',{a,b});
        end
        
        % Overloading a-b
        function r = minus(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'-',{a,b});
        end
        
        % Overloading -x
        function r = uminus(a)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'u-',{a});
        end
        
        % Overloading a*b
        function r = mtimes(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if a == b
                r = a^2;
            else
                r = matexp([],'*',{a,b});
            end
        end
        
        % Overloading a.*b
        function r = times(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'.*',{a,b});
        end
        
        % Overloading a'
        function r = ctranspose(a)
            r = matexp([],'transpose',{a});
        end
        
        % Overloading a.^b
        function r = power(a,b)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            r = matexp([],'power',{a,b});
        end
        
        % Overloading of accesso a(...) BUT only a(:) supported for
        % flattening/vec
        function r = subsref(a,S)
            assert(strcmp(S.type,'()'),'Only (:) supported');
            assert(iscell(S.subs) & numel(S) == 1,'Only (:) supported');
            assert(strcmp(S.subs{1},':'),'Only (:) supported');
            r = matexp([],'vec',{a});
        end
        
        % Overloading a*b
        function r = mpower(a,b)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            r = matexp([],'mpower',{a,b});
        end
        
        function r = exp(a)
            r = matexp([],'exp',{a});
        end
        
        function r = trace(a)
            r = matexp([],'trace',{a});
        end
        
        function r = inv(this)
            r = matexp([],'inv',{this});
        end
        
        function r = det(this)
            r = matexp([],'det',{this});
        end
        
        function r = diag(this)
            r = matexp([],'diag',{this});
        end
        
        function r = kron(a,b)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            r = matexp([],'kron',{a,b});
        end
        
        function r = log(this)
            assert(strcmp(this.aop,'det'),'Only log det supported');
            r = matexp([],'logdet',{this.aoperands{1}});
        end
        
        % make column vector
        function r = vec(this)
            r = matexp([],'vec',{this});
        end
        
        % make column vector
        function r = cos(this)
            r = matexp([],'cos',{this});
        end
        
        function r = sin(this)
            r = matexp([],'sin',{this});
        end
        
        function r = size(this)
            r = size(this.avalue);
        end
        
        function r = numel(this)
            r = numel(this.avalue);
        end
        
        % returns the value
        function r = value(this)
            r = this.avalue;
        end
        
        % name for variables matexp
        function r = name(this)
            r = this.aname;
        end
        
        % returns adjoint
        function r = adjoint(this)
            r = this.aadjoint;
        end
        
        % INTERNAL: increments adjoint
        function this = incadjoint(this,value)
            this.aadjoint = this.aadjoint + value;
        end
        
        % sets the value for the constant or variable ones
        function set(this,value)
            assert(isempty(this.aop));
            this.avalue = value;
        end
        
        % sets the value for the constant or variable ones
        function setadjoint(this,a)
            this.aadjoint = a;
        end
        
        % used in Second Order. Transforms the expression into a flatten
        % list with two arrays: 
        %
        %   r = { allops, 2 }
        %
        % First column contains the matexp, second column the list of
        % depending variables as indices in r
        %
        % c = [ops,vars] as counters
        function [r,c] = flatten(this)
            flatten_clear(this);
            % count all expressions and variables
            c = flatten_count(this,[0,0]);
            
            % prepare output cell
            Phi = cell(c(1)+c(2),2);
            % wrap in class (or in matexp)
            tPhi = matexp(Phi);
            flatten_fill(this,tPhi);
            % emit reesult with flattened cell
            r = tPhi.avalue;
        end
        
        function [H,v] = hessian(this)
            [r,c] = flatten(this);
            [H] = matexp.hessianpush(r,c);
            v = r(c(1)+1:end,1);
        end
    end
    methods (Access=private)
        % Used internally for filling in the flattened form
        function flatten_fill(this,tPhi)
            if this.aadjoint == 0
                return
            end
            ni = this.aadjoint;
            if ni < 0
                ni = size(tPhi.avalue,1)+ni+1; % ni=-1 == last
            end
            tPhi.avalue{ni,1} = this;
            c = zeros(length(this.aoperands),1);
            for I=1:length(this.aoperands)
                % assign children identifiers
                tni = this.aoperands{I}.adjoint;
                if tni < 0
                    tni = size(tPhi.avalue,1)+tni+1;
                end
                c(I) = tni;
                flatten_fill(this.aoperands{I},tPhi);
            end
            tPhi.avalue{ni,2} = c;
        end
        
        % Used internally for preparing the flattening
        function flatten_clear(this)
            this.aadjoint = [];
            for I=1:length(this.aoperands)
                if ~isempty(this.aoperands{I}.aadjoint)
                    flatten_clear(this.aoperands{I});
                end
            end
        end
        
        % Used internally for counting variables
        function c  =  flatten_count(this,c)
            if isempty(this.aadjoint)
                % constant vs var
                if isempty(this.aop) & isempty(this.aname) == 0
                    c(2) = c(2) + 1;
                    this.aadjoint = -c(2);
                else
                    % MAYBE excluded from the flattening BUT included in the
                    % three
                    %if isempty(this.aop)
                    %    this.aadjoint = 0;
                    %    return;
                    %end
                    c(1) = c(1) + 1;
                    this.aadjoint = c(1);
                    for I=1:length(this.aoperands)
                        c = flatten_count(this.aoperands{I},c);
                    end
                end
            end
        end
        
        % J(this,*) computes all the partial derivatives
        function r = parder(this)
            ops = this.aoperands;
            V = this.avalue; % this value
            
            % scalar functions f(X) =>  diag(vec(df(X)))
            switch(this.aop)
                case {'+','-'}
                    % this is trivial except for the case of scalar
                    [Al,Ar] = matexp.dsum(ops{1}.avalue,ops{2}.avalue,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Ar) & this.aop == '-'
                        Ar = -Ar;
                    end
                    r = {Al,Ar};                    
                case 'u-'
                    r = {-1};
                case 'exp'
                    assert(length(V)==1,'scalar only');
                    r = {V};                    
                case '.*'
                    [Al,Ar] = matexp.dsmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    r = {Al,Ar};
                case '*'
                    [Al,Ar] = matexp.dmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    r = {Al,Ar};    
                case 'cos'
                    q = sin(ops{1}.avalue);
                    r = {-diag(q(:))};
                case 'sin'
                    q = cos(ops{1}.avalue);
                    r = {diag(q(:))};
                case 'kron'
                    %assert(size(A,2)==numel(ops{1}.avalue)*numel(ops{2}.avalue),'kron input adjoint');
                    sA = [NaN,numel(ops{1}.avalue)*numel(ops{2}.avalue)];
                    [Al,Ar] = matexp.dkron(sA,ops{1}.avalue,ops{2}.avalue,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    r = {Al,Ar};
                case 'power'
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    assert(numel(ops{2}.avalue) == 1,'power needs to be scalar');
                    nexp = ops{2}.avalue;
                    switch nexp
                        case 1
                            % X^1 == X
                            r = {1,[]};
                        case 2
                            % X.^2 scalar op
                            r = {2*diag(ops{1}.avalue(:)),[]};
                        otherwise
                            r = {nexp*diag(ops{1}.avalue(:).^(nexp-1)),[]};
                    end
                case 'mpower'
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    if length(ops{1}.avalue) == 1
                        k = ops{2}.avalue;
                        r = {k*ops{1}.avalue^(k-1),[]};
                    else
                        switch ops{2}.avalue
                            case 1
                                r = {1,[]};
                            case -1
                                r = {-kron(V,V'),[]};
                            case 2
                                X = ops{1}.avalue;
                                Q = eye(length(X));
                                r = {(kron(Q,X)+kron(X',Q)),[]};
                            case 3
                                X = ops{1}.avalue;
                                r = {(kron((X')^2,eye(length(X)))+kron(X',X)+kron(eye(length(X)),X^2)),[]};
                            otherwise
                                error('not implemented generic matrix power exponent');%
                        end
                    end
                case 'det'
                    q = inv(value(ops{1}))';
                    r = {V*q(:)'};
                case 'log'
                    assert(strcmp(ops{1}.aop,'det'),'Only log det supported');
                    r = {1.0/det(ops{1}.avalue)};
                case 'logdet'
                    q = inv(value(ops{1}))';
                    r = {q(:)'};
                case 'trace'
                    q = (eye(length(ops{1}.avalue)));  % was colum(.)'
                    r = {q(:)'};
                case 'inv'
                    % S version for trace: vec'(A) (-kron(V,V')) = vec'(-VAV)
                    % in jfd.pdf there is no negative sign
                    % for dmb it should be: - kron(V',V)
                    r = {-kron(V',V)};
                case 'transpose'
                    % A is [kl, mn] where kl is the final output
                    % V is [mn, mn]
                    % we need to apply a permutation matrix that flips the
                    % mn so that we flip the output
                    % From dmb this is called TVEC: Tm,n = TVEC(m,n) is the vectorized transpose matrix
                    
                    % X' = unvec(permuterows(vec(X)))
                    % vec(X') = TVEC(m,n) vec(X)
                    % vec(X) = A[nm,m] X[m,n] B[n,1]    ?
                    % unvec(X) = A[m,nm] X[mn,1] B[1,n] ?
                    % i,jth element is 1 if j=1+m(i-1)-(mn-1)floor((i-1)/n)
                    
                    % Taken from: http://www.mathworks.com/matlabcentral/fileexchange/26781-vectorized-transpose-matrix/content/TvecMat.m
                    % note V is the transpose
                    Tmn = matexp.dtranspose(V);
                    r={Tmn};
                case 'vec'
                    % expression from [m,n] to [mn,1]
                    % adjoint receives [outsize,mn]
                    r={1};
                case 'diag'
                    % expression from [q,1] to [q,q]
                    % adjoint receives [outsize,qq] to [q,1] via a matrix [qq,q]
                    %
                    n = length(V);
                    q = zeros(n*n,n);
                    % OPTIMIZE ME
                    J = 1;
                    for I=1:n
                        q(J,I) = 1;
                        J = J +n+1;
                    end
                    r ={q};
                case ''  % nothing
                    r = {1};
                    return
                otherwise
                    error(['Unimplemented ' this.aop]);
            end
            
        end
        
        % J(this,j,k) with j,k as indices in operands, eventually using the
        % previos partial derivatives
        function r = parder2(this,j,k,phide)
            ops = this.aoperands; % operands
            V = this.avalue; % this value
            J = phide;       % cell array of all partial derivatives
            
            % scalar functions f(X) =>  diag(vec(df(X)))
            switch(this.aop)
                case {'+','-'}
                    r = 0;
                case 'u-'
                    r = 0;
                case 'exp'
                    assert(length(V) == 1,'only scalar for exp');
                    r = V;                    
                case '.*'
                    error('scalar produce not implemented');
                case '*'
                    % Operation: L*R
                    %
                    % Jl = kron(R',eye(nl));    
                    %   kron size is: [nr*nl, common*nl]
                    % Jr = kron(eye(nr),L);
                    %   kron size is: [nr*nl,nr*common]
                    % Jlr = J(kron(R',eye(nl),R)
                    %   [nr*nl,common*nl,common*nr]
                    % Jrl = J(kron(eye(nr),L),L)
                    %   [nr*nl,common*nl,common*nr]
                    % nl=3
                    % common=5
                    % nr=2;
                    % We could use dkron flat (tested) or the tensor
                    % version
                    
                    nl = size(V,1);
                    nr = size(V,2);            
                    if k == j 
                        % V=L*R
                        % V=kron(R',eye(nl)) -- kron(eye(nr),L)
                        % diff(V,L)
                        % diff(V,R)
                        r = 0;
                    else                                                
                        if length(ops{2}.avalue) == 1 == length(ops{1}.avalue) == 1
                            % special case?
                            if ops{2} == ops{1}
                                r = 2;
                            else
                                r = 1;
                            end
                        else
                            if j == 1 % k==2       
                                % J(kron(R',eye(nl)),R) as matrix
                                r = matexp.dkronR(ops{2}.avalue,nl,ops{2}.avarcount > 0);
                                assert(all(size(r) == [nl*nr,numel(ops{1}.avalue),numel(ops{2}.avalue)]));
                            else % j == 2, k == 1
                                % J(kron(eye(nr),L),L) as matrix
                                r = matexp.dkronL(ops{1}.avalue,nr,ops{1}.avarcount > 0);
                                assert(all(size(r) == [nl*nr,numel(ops{2}.avalue),numel(ops{1}.avalue)]));
                            end
                        end
                    end
                    
                case 'cos'
                    error('not implemented, use general case of scalar dependency');
                    r = 0;
                case 'sin'
                    error('not implemented, use general case of scalar dependency');
                    r = 0;
                case 'kron'
                    error('will never implement');
                case 'power' % scalar power .^
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    assert(numel(ops{2}.avalue) == 1,'power needs to be scalar');
                    % [2,*] = 0
                    % [*,2] = 0
                    % [1,1] first order is dimension L (rows*cols), second order is [L,L] each element in first diagonal
                    %   becomes one more
                    if j == 2 || k == 2
                        r = 0;
                    else
                        k = ops{2}.avalue;
                        switch k
                            case 1
                                % X => 1 => 0
                                r = 0;
                            otherwise
                                q = numel(V);
                                Q = (k*(k-1))*(ops{1}.avalue).^(k-2);
                                r = zeros(q,q*q,'like',V);
                                r(sub2ind(size(r),1:q,((1:q)-1)*(q+1)+1)) = Q(:);
                                %ALT with for
                                %Q = Q(:);
                                %for I=1:length(Q)
                                %    r(I,(I-1)*(numel(V)+1)+1) = Q(I);
                                %end
                                %
                                %ALT Tensor version
                                %for I=1:length(Q)
                                %    r(I,I,I) = Q(I);
                                %end
                        end
                    end
                case 'mpower'           
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    % scalar_f(x)^k
                    % [1 1] k (k-1) X^(k-2)
                    % [1 2] 0
                    % [2 1] 0
                    % [2 2] 0
                    if j == 2 || k == 2
                        r = 0;
                    else 
                        k = ops{2}.avalue;
                        r = k*(k-1)*ops{1}.avalue^(k-2);
                    end
                case 'det'
                    error('unimplemented');
                case 'log'
                    error('unimplemented');
                case 'logdet'
                    error('unimplemented');
                case 'trace'
                    r = 0;
                case 'inv'
                    error('unimplemented');
                case 'transpose'
                    r = 0;
                case 'vec'
                    r = 0;
                case 'diag'
                    r = 0;
                case ''  % nothing
                    r = 0;
                    return
                otherwise
                    error(['Unimplemented ' this.aop]);
            end
            
        end
        
        % support function of the recursive AD
        function mautodiff(this)
            
            ops = this.aoperands;
            A = this.aadjoint;
            V = this.avalue; % this value
            
            % scalar functions f(X) =>  diag(vec(df(X)))
            switch(this.aop)
                case {'+','-'}
                    % this is trivial except for the case of scalar
                    [Al,Ar] = matexp.dsum(ops{1}.avalue,ops{2}.avalue,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                    end
                    if ~isempty(Ar)
                        if this.aop == '-'
                            Ar = -Ar;
                        end
                        incadjoint(ops{2},A*Ar);
                    end
                case 'exp'
                    assert(length(V)==1,'exp only scalar');
                    incadjoint(ops{1},A*V);
                case 'u-'
                    incadjoint(ops{1},-A);
                case '.*'
                    [Al,Ar] = matexp.dsmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                    end
                    if ~isempty(Ar)
                        incadjoint(ops{2},A*Ar);
                    end
                case '*'


                    [Al,Ar] = matexp.dmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    % L R   [nl,q] [q,nr] -> [nl,nr]
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                    end
                    if ~isempty(Ar)
                        incadjoint(ops{2},A*Ar);
                    end
                case 'cos'
                    q = sin(ops{1}.avalue);
                    incadjoint(ops{1},-A*diag(q(:)));
                case 'sin'
                    q = cos(ops{1}.avalue);
                    incadjoint(ops{1},A*diag(q(:)));
                case 'kron'
                    assert(size(A,2)==numel(ops{1}.avalue)*numel(ops{2}.avalue),'kron input adjoint');
                    [Al,Ar] = matexp.dkron(size(A),ops{1}.avalue,ops{2}.avalue,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                    end
                    if ~isempty(Ar)
                        incadjoint(ops{2},A*Ar);
                    end
                case 'power'
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    assert(numel(ops{2}.avalue) == 1,'power needs to be scalar');
                    nexp = ops{2}.avalue;
                    switch nexp
                        case 1
                            % X^1 == X
                            incadjoint(ops{1},A);
                        case 2
                            % X.^2 scalar op
                            incadjoint(ops{1},2*A*diag(ops{1}.avalue(:)));
                        otherwise
                            incadjoint(ops{1},nexp*A*diag(ops{1}.avalue(:).^(nexp-1)));
                    end
                case 'mpower'
                    assert(ops{2}.avarcount == 0,'power needs to be constant');
                    if length(ops{1}.avalue) == 1
                        k = ops{2}.avalue;
                        incadjoint(ops{1},A*k*ops{1}.avalue^(k-1));
                    else
                        switch ops{2}.avalue
                            case 1
                                incadjoint(ops{1},A);
                            case -1
                                incadjoint(ops{1},-A*kron(V,V'));
                            case 2
                                X = ops{1}.avalue;
                                Q = eye(length(X));
                                incadjoint(ops{1},A*(kron(Q,X)+kron(X',Q)));
                            case 3
                                X = ops{1}.avalue;
                                incadjoint(ops{1},A*(kron((X')^2,eye(length(X)))+kron(X',X)+kron(eye(length(X)),X^2)));
                            otherwise
                                error('not implemented generic matrix power exponent');%
                        end
                    end
                case 'det'
                    q = inv(value(ops{1}))';
                    incadjoint(ops{1},A*V*q(:)');
                case 'log'
                    assert(strcmp(ops{1}.aop,'det'),'Only log det supported');
                    incadjoint(ops{1},A/det(ops{1}.avalue)); % log det (X) = (X^-1'):'
                case 'logdet'
                    q = inv(value(ops{1}))';
                    incadjoint(ops{1},A*q(:)'); % log det (X) = (X^-1'):'
                case 'trace'
                    q = (eye(length(ops{1}.avalue)));  % was colum(.)'
                    incadjoint(ops{1},A*q(:)');
                case 'inv'
                    % S version for trace: vec'(A) (-kron(V,V')) = vec'(-VAV)
                    % in jfd.pdf there is no negative sign
                    % for dmb it should be: - kron(V',V)
                    incadjoint(ops{1},-A*kron(V',V));
                case 'transpose'
                    % A is [kl, mn] where kl is the final output
                    % V is [mn, mn]
                    % we need to apply a permutation matrix that flips the
                    % mn so that we flip the output
                    % From dmb this is called TVEC: Tm,n = TVEC(m,n) is the vectorized transpose matrix
                    
                    % X' = unvec(permuterows(vec(X)))
                    % vec(X') = TVEC(m,n) vec(X)
                    % vec(X) = A[nm,m] X[m,n] B[n,1]    ?
                    % unvec(X) = A[m,nm] X[mn,1] B[1,n] ?
                    % i,jth element is 1 if j=1+m(i-1)-(mn-1)floor((i-1)/n)
                    
                    % Taken from: http://www.mathworks.com/matlabcentral/fileexchange/26781-vectorized-transpose-matrix/content/TvecMat.m
                    % note V is the transpose
                    Tmn = matexp.dtranspose(V);
                    incadjoint(ops{1},A*Tmn);
                case 'vec'
                    % expression from [m,n] to [mn,1]
                    % adjoint receives [outsize,mn]
                    incadjoint(ops{1},A);
                case 'diag'
                    % expression from [q,1] to [q,q]
                    % adjoint receives [outsize,qq] to [q,1] via a matrix [qq,q]
                    %
                    n = length(V);
                    q = zeros(n*n,n);
                    % OPTIMIZE ME
                    J = 1;
                    for I=1:n
                        q(J,I) = 1;
                        J = J +n+1;
                    end
                    incadjoint(ops{1},A*q);
                case ''  % nothing
                    return
                otherwise
                    error(['Unimplemented ' this.aop]);
            end
            this.aoperands = ops;
            
            % then continue the descent ONLY if meaningful
            for I=1:length(this.aoperands)
                if this.aoperands{I}.avarcount > 0
                    mautodiff(this.aoperands{I});
                end
            end
        end
    end
    methods(Static)
        % computes 1st and 2nd using the push edge applied over the flat
        % version of the algorithm.
        %
        % Requirements: full function evaluated, full funciton flattened
        % [r,c] = flatten(exp)
        % r = cell [exps;children]
        % c = [countops,countvars]
        %
        % J is returned inside the adjoint of every variable as in regular
        % case, H is returned separately in cell form (TBD matrix form)
        %
        % sout = topmost output flattened
        % sin  = sum numel(input)
        % Hessian is [sout,sin,sin] that is [outrow,outcol,
        % [inrow_i,incol_i]_i,...]
        %
        % Note: due to the use of tprod we cannot use symbolic (!)
        function H = hessianpush(r,c)
            l = c(1);
            n = c(2);
            Wf = zeros(l+n,l+n);
            W  = num2cell(zeros(l+n,l+n));
            v = cell(l+n,1);
            sout = numel(r{1,1}.avalue); % output of topmost
            for i=2:l+n
                v{i} = 0;
            end
            v{1} = eye(sout);
            % In the paper intermediates are reversed and start from top
            for i=1:l  
                chi = r{i,2}; % list of children indices
                % push only in subsequent
                sm = find(Wf(:,i)); % non zero descendents or self (maybe Wf(:,i))
                %sm = 1:length(Wf); % WF above IS BUGGY
                sm = sm(sm >= i); % p >= i (descendent)
                phide = parder(r{i,1});
                for ip=1:length(sm)
                    p = sm(ip);
                    if p ~= i
                        for ij=1:length(chi)
                            j = chi(ij);
                            if isempty(phide{ij})
                                continue
                            end
                            % [Ll,Lj,Lp] += [Ll,Lp,Li] *?* [Li,Lj] = OK
                            % (note reorder jp)
                            % ONLY store row => col
                            % W{p,i}  [Ll,Lp,Li]
                            % phide{ij} [Li,Lj]
                            % p > i                            
                            if ismatrix(W{p,i})
                                w = W{p,i}*phide{ij};
                                {'push',p,j}
                                if p == j
                                    W{p,p} = W{p,p} + 2 * w;
                                    Wf(p,p) = 1;
                                elseif p > j
                                    W{p,j} = W{p,j} + w;
                                    Wf(p,j) = 1;
                                else
                                    W{j,p} = W{j,p} + w;
                                    Wf(j,p) = 1;
                                end
                            else
                                error('new version without tensors')
                                w = tprod(W{p,i},[1,2,-1],phide{ij},[-1,3]);
                                if p == j
                                    W{p,p} = W{p,p} + 2 * w;
                                    Wf(p,p) = 1;
                                elseif p > j
                                    W{p,j} = W{p,j} + w;
                                    Wf(p,j) = 1;
                                else % j > p
                                    W{j,p} = W{j,p} + w;
                                    Wf(j,p) = 1;
                                end
                            end
                        end
                    else % p == i
                        % all unordered pairs (j and k > i any way)
                        for ij=1:length(chi)
                            j = chi(ij);
                            if isempty(phide{ij})
                                continue
                            end
                            for ik=ij:length(chi)    % can't be 
                                k = chi(ik);
                                if isempty(phide{ik})
                                    continue
                                end
                                % Ll is size ouf output
                                % [Ll,Lk,Lj] += [Ll,Li,Li] * [Li,Lj]
                                % *?* [Li,Lk] = [L,Li,Lj] *?* [Li,Lk] =
                                % special tensor product
                                % ALWAYS j>=k
                                if ismatrix(W{i,i})
                                    w = ((W{i,i}*phide{ij})'*phide{ik})';                                    
                                else
                                    error('new version without tensors')
                                    w = tprod(tprod(W{i,i},[1,2,-1],phide{ij},[-1,3]),[1,-1,2],phide{ik},[-1,3]);
                                end
                                if k > j
                                    {'pushbi',k,j}
                                    W{k,j} = W{k,j} + w;
                                    Wf(k,j) = 1;
                                else
                                    {'pushbi',j,k}
                                    W{j,k} = W{j,k} + w;
                                    Wf(j,k) = 1;
                                end
                            end
                        end
                    end
                end
                % create
                % all unordered pairs: binary is (1,2) (1,1) (2,1)
                % 
                
                for ij=1:length(chi)
                    j = chi(ij);
                    for ik=ij:length(chi)    % FOR TESTING WE HAVE
                        k = chi(ik);
                        pd2 = parder2(r{i,1},ij,ik,phide);
                        if length(pd2) == 1 && pd2 == 0
                            % zero
                        else
                            if ismatrix(pd2)
                                w = v{i}*pd2;
                            else
                                error('new version without tensors')
                                % TENSORS not admitted in sym
                                w = tprod(v{i},[1,-1],pd2,[-1,2,3]);
                            end
                            if j == k
                                    %{'create',j,j}
                                    %r{i,1}
                                    %pd2
                                W{j,j} = W{j,j} + w;
                                Wf(j,j) = 1; %any(w(:) ~= 0);
                            elseif j > k
                                    {'create',j,k}
                                W{j,k} = W{j,k} + w;
                                Wf(j,k) = 1; %any(w(:) ~= 0);
                            else
                                    {'create',k,j}
                                W{k,j} = W{k,j} + w;
                                Wf(k,j) = 1; %any(w(:) ~= 0);
                            end
                        end
                    end
                end
                
                % adjoint: parent by children par der
                for ij=1:length(chi)
                    j = chi(ij);
                    if ~isempty(phide{ij})
                        v{j} = v{j} + v{i}*phide{ij}; % relative child is inside exp
                    end
                end
            end
            % extract J and H
            H = W(l+1:end,l+1:end); % symmetric with empty=0
            % assign adjoint to each target variable
            for i=l+1:length(v)
                r{i,1}.setadjoint(v{i});
            end
            
        end
        
        % Compute the column version form of transposition. That is if we
        % have a matrix X [m,n] flattened X: [mn,1] and we want to compute
        % the transpose of X in flattened form (X':) we can apply the
        % matrix Tmn for this purpose:  (X':) == Tmn X:
        function Tmn = dtranspose(V)
            n = size(V,1);
            m = size(V,2);
            d = m*n;
            Tmn = zeros(d,d);
            
            i = 1:d;
            rI = 1+m.*(i-1)-(m*n-1).*floor((i-1)./n);
            I1s = sub2ind([d d],rI,1:d);
            Tmn(I1s) = 1;
            Tmn = Tmn';
        end
        
        % specialized derivative of kron(eye(nr),L) in Tensor Form
        function Ar = dkronRT(R,nl,rnc)
            error('replaced tensor with matrix flattening');
            k2 = size(R,1);
            k1 = size(R,2);
            k3 = nl;
            if rnc
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(k1*k3,k3*k2,k1*k2,'like',R);
                t = eye(k3);
                % vertical by
                k = 1;
                for J=1:k3:size(q,1)
                    for I=1:k3:size(q,2)
                        q(J:J+k3-1,I:I+k3-1,k) = t;
                        k = k + 1;
                    end
                end
                Ar = q;
            else
                Ar = [];
            end
        end
        
        % specialized derivative of kron(kron(eye(nl),L),L) in Tensor Form
        function Al = dkronLT(L,nr,lnc)
            %error('replaced tensor with matrix flattening');
            k1 = size(L,1);
            k2 = size(L,2);
            k3 = nr;
            if lnc
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(k1*k3,k2*k3,k1*k2,'like',L);
                t = eye(k1);
                % vertical by
                k = 1;
                for J=1:k1:size(q,1)  % == k3
                    for I=1:k1:size(q,3) % == k2
                        q(J:J+k1-1,k,I:I+k1-1) = t;
                        k = k + 1;
                    end
                end
                Al = q;
            else
                Al = [];
            end
        end
        
        
        % specialized derivative of kron(kron(eye(nl),L),L) in Matrix Form
        % [topfunction,R size,L size] => [topfunction, Rsize stack Lsize]
        function Al = dkronL(L,nr,lnc)
            k1 = size(L,1);
            k2 = size(L,2);
            k3 = nr;
            if lnc
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                ss = [k2*k3,k1*k2];
                q = zeros(k1*k3,ss(1)*ss(2),'like',L);
                %
                % repeat k2 times (shifting right by k1*k3*k2)
                %   repeat k1 times (shifting down by 1)
                %       total items k3:
                %           horizontal by k2
                %           vertical by k1
                for J=1:k2
                    ox = k1*k3*k2*(J-1);
                    for I=1:k1
                        oy = I-1;
                        q(sub2ind(size(q),1+oy+((1:k3)-1)*k1,1+ox+((1:k3)-1)*k2)) = 1;
                    end
                end
                        
                Al = q;
            else
                Al = [];
            end
        end
        
        % specialized derivative of kron(eye(nr),L) in Matrix Form
        % [topfunction,L size,R size] => [topfunction, Lsize stack Rsize]
        function Ar = dkronR(R,nl,rnc)
            k2 = size(R,1);
            k1 = size(R,2);
            k3 = nl;
            if rnc
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(k1*k3,k3*k2 * k1*k2,'like',R);
                t = eye(k3,'like',R);
                
                % inner loop: k2 times shifting k1*k1*k3 hor
                % outer loop: k3 times shifting of k1 ver
                
                for J=1:k1
                    oy = (J-1)*k3;
                    for I=1:k2
                        ox = (I-1)*k1*k1*k3;
                        q(oy+(1:k3),ox+(1:k3)) = t;
                    end
                end
                Ar = q;
            else
                Ar = [];
            end
        end
        
                
        
        % helper for the kronecker product for the use in other cases
        % NOTE: this should be optimized in a way to transform it into some form of product and not picking KI
        %   e.g. it could be a pair of indices (source,target)
        function [Al,Ar] = dkron(sA,L,R,lnc,rnc)
            % kron(A,B) = P kron(B,A) Q
            % kron(A,B) means expand
            if lnc
                % kronout = numel(L)*numel(R)
                % adjoint A=[outsize, kronout] X=[kronout, numel(R)]
                %      lr lc rr rc,   ll lc ==
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(sA(2),numel(L),'like',R);
                q0 = zeros(size(R,2)*size(L,1)*size(R,1),size(L,1),'like',R);
                
                for I=1:size(R,2)
                    for J=1:size(L,1)
                        for K=1:size(R,1)
                            q0((I-1)*size(R,1)*size(L,1)+(J-1)*size(R,1)+(K-1)+1,(J-1)+1) = R(K,I);
                        end
                    end
                end
                % replicate
                for I=1:size(L,2)
                    q((I-1)*size(q0,1)+1:I*size(q0,1),(I-1)*size(L,1)+1:I*size(L,1)) = q0;
                end
                Al = q;
            else
                Al = [];
            end
            if rnc
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(sA(2),numel(R),'like',L);
                bx = size(R,1);
                by = size(R,1)*size(L,1)*size(R,2);
                for I=1:size(R,2) % along x: each is size(R,1)
                    for J=1:size(L,2) % along y: each is: size(R,1)*size(L,1)*size(R,2)
                        for K=1:size(L,1) % long y: each is size(R,1)
                            ki = (J-1)*by+1+(K-1)*bx+(I-1)*bx*size(L,1);
                            q(ki:ki+bx-1,(I-1)*bx+1:(I-1)*bx+bx) = L(K,J)*eye(bx);
                        end
                    end
                end
                Ar = q;
            else
                Ar = [];
            end
        end
        
        
        % helper for the matrix product dealing also with scalar expansion
        % Return empty Al,Ar when constant
        function [Al,Ar] = dmul(L,R,V,lnc,rnc)
            nl = size(V,1);
            nr = size(V,2);
            
            %Note for fast version:
            % LEFT:  vec'(A)(I kron R')   = vec'(R A I)
            % RIGHT: vec'(A)(L kron I )   = vec'(I A L)
            % kron([a,b],[c,d]) is [ac,bd]
            if lnc
                % L is scalar, R is not => R output
                if length(L) == 1 & length(R) > 1
                    % R is [nl,nr]
                    % d y/dx = [a b11  a b12; a b21 a b22; a b31
                    % ab32] = kron(nl,R) * [nl nl, nl nr] [nl*nr,1 == kron(nl,1,nr,1)] a/dx
                    %
                    Al = diag(R(:))*ones(nl*nr,1);
                elseif length(L) > 1 & length(R) == 1
                    Al = R*eye(numel(L));
                else
                    % D Al = d kron(R', eye(nl)) = ... d(R')
                    Al = kron(R',eye(nl));
                end
            else
                Al = [];
            end
            if rnc
                % right is scalar, left is not => output is left sized
                if length(R) == 1 & length(L) > 1
                    Ar = diag(L(:))*ones(nl*nr,1);
                    % left is scalar, right is not => output is right sized
                elseif length(R) > 1 & length(L) == 1
                    % L is scalar, output is R dominated
                    Ar = L*eye(numel(R));
                    % regular product
                else
                    Ar = kron(eye(nr),L);
                end
            else
                Ar = [];
            end
        end
        
        % helper for the memberwise product with also scalar expansion
        function [Al,Ar] = dsmul(L,R,V,lc,rc)
            % L R
            
            nl = size(V,1);
            nr = size(V,2);
            if lc
                if length(L) == 1 & length(R) > 1
                    % enforce enlarge L
                    Al = diag(R(:))*ones(nl*nr,1);
                else
                    Rt = R';
                    Al = diag(Rt(:));
                end
            else
                Al = [];
            end
            if rc
                if length(R) == 1 & length(L) > 1
                    % enforce enlarge R
                    Ar = diag(L(:))*ones(nl*nr,1);
                else
                    Ar = diag(L(:));
                end
            else
                Ar = [];
            end
        end
        
        % helper for the memberwise sum
        function [Al,Ar] = dsum(L,R,lc,rc)
            if lc
                Al = 1;
            else
                Al = [];
            end
            if rc
                Ar = 1;
            else
                Ar = [];
            end
            
            if length(L) == 1
                if length(R) == 1
                    % both -> do nothing
                else
                    % L is scalar: propagate size of R to L, that
                    % is: column(eye(numel(R),value=L))
                    % A is [outsize, numel(R)]
                    % [outsize, numel(R)] * [numel(R), 1]
                    Al = ones(numel(R),1);
                end
            elseif length(R) == 1
                % R is scalar: propagate size of L to R
                Ar = ones(numel(L),1);
            end
        end
        
    end
end

