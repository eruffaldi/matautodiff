% Matrix Expression Class for Automatic Differentiation
% Emanuele Ruffaldi 2016, Scuola Superiore Sant'Anna
%
% Provides Reverse-Mode Automatic Differentiation of Matrices
classdef matexp < handle
    
    properties
        aname
        avalue
        aadjoint
        ahessian
        aop
        aoperands
        avarcount
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
        
        function autodiff(this,c2)
            if nargin == 1
                c2 = 0;
            end
            if isempty(this.avalue)
                update(this);
            end
            resetadjoint(this,0);
            % [k,m] -> [k,m] => eye [k,m,k,m] => [km,km] jacobian matrix
            this.aadjoint = eye(numel(this.avalue)); 
            if c2 
                % [k,m] -> [k,m] => [k,m,k,m,k m] => [km,km*km] hessian
                this.ahessian = 1; 
            end
            mautodiff(this,c2);
        end    
        
       
        % final hessian tensor: [k,l,m,n,m,n] stacked as [kl,mn mn] 
        % but at start we don't know [mn]
        %this.ahessian = eye(numel(this.avalue));
        
        % collects the variables present in the expression tree
        function r = collectvars(this)
            r = ucollectvars(this);
            % TODO unique
        end
        
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
            this.ahessian = x;
            this.aadjoint = x;
        end

        % updates the value
        function update(this)
            for I=1:length(this.aoperands)
                update(this.aoperands{I});
            end
            switch(this.aop)
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
                
        function r = plus(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'+',{a,b});
        end        
        
        function r = minus(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'-',{a,b});
        end        
        
        function r = uminus(a)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'u-',{a});
        end        
        
        function r = mtimes(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'*',{a,b});
        end        
        
        function r = times(a,b)
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            r = matexp([],'.*',{a,b});
        end        
        
        function r = ctranspose(a)
            r = matexp([],'transpose',{a});
        end        
        

        function r = power(a,b)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            r = matexp([],'power',{a,b});
        end      
        
        function r = subsref(a,S)
            assert(strcmp(S.type,'()'),'Only (:) supported');
            assert(iscell(S.subs) & numel(S) == 1,'Only (:) supported');
            assert(strcmp(S.subs{1},':'),'Only (:) supported');
            r = matexp([],'vec',{a});
        end
        
        function r = mpower(a,b)
            if ~isa(a,'matexp')
                a = matexp(a);
            end
            if ~isa(b,'matexp')
                b = matexp(b);
            end
            r = matexp([],'mpower',{a,b});
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
        
        % size of the value
        function r = size(this)
            r = size(this.avalue);
        end
        
        % returns the value
        function r = value(this)
            r = this.avalue;
        end
        
        % name for variables
        function r = name(this)
            r = this.aname;
        end
        
        % returns adjoint
        function r = adjoint(this)
            r = this.aadjoint;
        end
       
        % returns adjoint
        function r = hessian(this)
            r = this.ahessian;
        end

        function this = incadjoint(this,value)
            this.aadjoint = this.aadjoint + value;
        end
        
        function this = inchessian(this,value)
            this.ahessian = this.ahessian + value;
        end

        % sets the value for the constant or variable ones
        function set(this,value)
            assert(isempty(this.aop));
            this.avalue = value;
        end         
    end
    methods (Access=private)

        function mautodiff(this,c2)
            
            ops = this.aoperands;
            A = this.aadjoint;
            H = this.ahessian;
            V = this.avalue; % this value
            
            % scalar functions f(X) =>  diag(vec(df(X)))
            switch(this.aop)
                case {'+','-'}
                    % this is trivial except for the case of scalar
                    [Al,Ar,Hl,Hr] = matexp.dsum(ops{1}.avalue,ops{2}.avalue,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                        if c2
                            inchessian(ops{1},H*Hl);
                        end
                    end
                    if ~isempty(Ar)                       
                        if this.aop == '-'
                            Ar = -Ar;
                            Hr = -Hr;
                        end
                        incadjoint(ops{2},A*Ar);
                        if c2
                            inchessian(ops{2},H*Hr);
                        end
                    end
                case 'u-'
                    incadjoint(ops{1},-A);
                    if c2
                        inchessian(ops{2},-H);
                    end
                case '.*'
                    [Al,Ar,Hl,Hr] = matexp.dsmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                        if c2
                            inchessian(ops{1},H*Hl);
                        end
                    end
                    if ~isempty(Ar)
                        incadjoint(ops{2},A*Ar);
                        if c2
                            inchessian(ops{2},H*Hr);
                        end
                    end
                case '*'
                    [Al,Ar,Hl,Hr] = matexp.dmul(ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
                    % L R   [nl,q] [q,nr] -> [nl,nr]
                    if ~isempty(Al)
                        incadjoint(ops{1},A*Al);
                        if c2
                            inchessian(ops{1},H*Hl);
                        end
                    end
                    if ~isempty(Ar)
                        incadjoint(ops{2},A*Ar);
                        if c2
                            inchessian(ops{2},H*Hr);
                        end
                    end
                case 'cos'
                    q = sin(ops{1}.avalue);
                    incadjoint(ops{1},-A*diag(q(:)));
                case 'sin'
                    q = cos(ops{1}.avalue);
                    incadjoint(ops{1},A*diag(q(:)));
                case 'kron'
                    assert(size(A,2)==numel(ops{1}.avalue)*numel(ops{2}.avalue),'kron input adjoint');
                    [Al,Ar] = matexp.dkron(A,ops{1}.avalue,ops{2}.avalue,V,ops{1}.avarcount > 0,ops{2}.avarcount > 0);
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
                    mautodiff(this.aoperands{I},c2);
                end
            end
        end    
    end
    methods(Static)
        % helper for the transposition, returns the Tmn matrix
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

        % helper for the kronecker product for the use in other cases
        % NOTE: this should be optimized in a way to transform it into some form of product and not picking KI
        %   e.g. it could be a pair of indices (source,target)
        function [Al,Ar] = dkron(A,L,R,V,lc,rc)
            % kron(A,B) = P kron(B,A) Q
            % kron(A,B) means expand 
            if lc          
                % kronout = numel(L)*numel(R)
                % adjoint A=[outsize, kronout] X=[kronout, numel(R)]
                %      lr lc rr rc,   ll lc == 
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(size(A,2),numel(L),class(R));
                q0 = zeros(size(R,2)*size(L,1)*size(R,1),size(L,1),class(R));
                
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
            if rc    
                % adjoint A=[outsize, kronout] X=[kronout, numel(L)]
                %      lr lc rr rc,   lr lc 
                % TODO finde: P kron(L,?) Q  with P,Q permutations
                q = zeros(size(A,2),numel(R),class(L));
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
        function [Al,Ar,Hl,Hr] = dmul(L,R,V,lc,rc)
            nl = size(V,1); 
            nr = size(V,2);
            Hl = [];
            hr = [];
            
            %Note for fast version: 
            % LEFT:  vec'(A)(I kron R')   = vec'(R A I) 
            % RIGHT: vec'(A)(L kron I )   = vec'(I A L)
            % kron([a,b],[c,d]) is [ac,bd]
            if lc
                if length(L) == 1 & length(R) > 1
                    % R is [nl,nr]
                    % d y/dx = [a b11  a b12; a b21 a b22; a b31
                    % ab32] = kron(nl,R) * [nl nl, nl nr] [nl*nr,1 == kron(nl,1,nr,1)] a/dx
                    %
                    Al = diag(R(:))*ones(nl*nr,1);   
                else
                    % D Al = d kron(R', eye(nl)) = ... d(R')
                    Al = kron(R',eye(nl));
                end
            else
                Al = [];
            end
            if rc
                %Ar = A*kron(L,eye(nr));
                Ar = kron(eye(nr),L);
                if length(R) == 1 & length(L) > 1
                    % L*eye(size(L,2))*r
                        Ar = diag(L(:))*ones(nl*nr,1);   
                end
                % l r   q r
            else
                Ar = [];
            end
        end

        % helper for the memberwise product with also scalar expansion
        function [Al,Ar,Hl,Hr] = dsmul(L,R,V,lc,rc)
            % L R
                Hl = [];
                Hr =[];
            
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
        function [Al,Ar,Hl,Hr] = dsum(L,R,lc,rc)
            if lc 
                Al = 1;
                Hl = 1;
            else
                Al = [];
                Hl = [];
            end
            if rc
                Ar = 1;
                Hr = 1;
            else
                Ar = [];
                Hr = [];
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

