% Matrix Expression Class for Automatic Differentiation
% Emanuele Ruffaldi 2016
%
% Provides Reverse-Mode Automatic Differentiation of Matrices
classdef matexp < handle
    
    properties
        aname
        avalue
        aadjoint
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

        function autodiff(this)
            if isempty(this.avalue)
                update(this);
            end
            % the root performs the reset of the adjoints
                resetadjoint(this,0);
                this.aadjoint = 1; 
                mautodiff(this);
                %if length(this.avalue) == 1
                % scalar path
                %resetadjoint(this,0);
                %this.aadjoint = 1; 
                %sautodiff(this);
        end
        
         % computes autodiff in recursive mode
        % ASSUMES an update of the value
        function mautodiff(this)
            
            ops = this.aoperands;
            A = this.aadjoint;
            V = this.avalue; % this value
            
            % scalar functions f(X) =>  diag(vec(df(X)))
            switch(this.aop)
                case '+'
                    incadjoint(ops{1},A);
                    incadjoint(ops{2},A);
                case 'u-'
                    incadjoint(ops{1},-A);
                case '-'
                    incadjoint(ops{1},A);
                    incadjoint(ops{2},-A);               
                case '.*'
                    % L R
                    L = ops{1}.avalue;
                    R = ops{2}.avalue;
                    Rt = R';
                    incadjoint(ops{1},A*diag(Rt(:)));
                    incadjoint(ops{1},A*diag(L(:)));                    
                case '*'
                    % L R
                    L = ops{1}.avalue;
                    R = ops{2}.avalue;
                    k = size(V,1);
                    l = size(V,2);
                    %Note for fast version: 
                    % vec'(A)(I kron R')=vec'(R A I) 
                    % vec'(A)(L kron I) = vec'(I A L)
                    %dmb ref:
                    % (R' kron I)
                    % (I kron L)
                    %incadjoint(ops{1},A*kron(eye(k),R')); % by derivative of op1
                    %incadjoint(ops{2},A*kron(L,eye(l))); % by derivative of op2
                    incadjoint(ops{1},A*kron(R',eye(k))); % by derivative of op1
                    incadjoint(ops{2},A*kron(eye(l),L)); % by derivative of op2
                case 'cos'
                    q = sin(ops{1}.avalue);
                    incadjoint(ops{1},-A*diag(q(:)));
                case 'sin'
                    q = cos(ops{1}.avalue);
                    incadjoint(ops{1},A*diag(q(:)));
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
                        case 2 
                            X = ops{1}.avalue;
                            Q = eye(length(X));
                            incadjoint(ops{1},A*(kron(Q,X)+kron(X',Q)));
                        case 3 
                            X = ops{1}.avalue;
                            incadjoint(ops{1},A*(kron((X')^2,eye(length(X)))+kron(X',X)+kron(eye(length(X)),X^2)));
                         otherwise
                            error('not implemented generic power');%                             
                     end
%                 case 'logdet'
%                     incadjoint(ops{1},inv(V)');
%                 case 'det'
%                     assert('Not implemented autodiff of det');
                case 'trace'  
                    incadjoint(ops{1},A*reshape(eye(length(ops{1}.avalue)),1,[]));
%                 case 'vec' % vectorization
                 case 'inv' 
                     % S version for trace: vec'(A) (-kron(V,V')) = vec'(-VAV)
                     % for jfd.pdf there is no negative sign ERROR
                     % for dmb it should be: - kron(V',V)
                     %
                     % note: kron is distributive against transposition =>
                     % transpose kron(V,V') == kron(V',V)
                     incadjoint(ops{1},-A*kron(V,V'));
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
                    n = size(V,1); 
                    m = size(V,2);
                    d = m*n;
                    Tmn = zeros(d,d);

                    i = 1:d;
                    rI = 1+m.*(i-1)-(m*n-1).*floor((i-1)./n);
                    I1s = sub2ind([d d],rI,1:d);
                    Tmn(I1s) = 1;
                    Tmn = Tmn';
                    
                    incadjoint(ops{1},A*Tmn);
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
%         
%         % special case when output is trace(F(X)) and we have no fully matrix
%         % operations (TO BE VERIFIED) corresponds to ADTalk.pdf and takes
%         % advantage of special properties of the trace
%         function sautodiff(this)
%             
%             ops = this.aoperands;
%             A = this.aadjoint;
%             V = this.avalue; % this value
%             
%             % by operation, increment adjoint of children
%             switch(this.aop)
%                 case '+'
%                     incadjoint(ops{1},A);
%                     incadjoint(ops{2},A);
%                 case '-'
%                     incadjoint(ops{1},A);
%                     incadjoint(ops{2},-A);
%                 case '*'
%                     incadjoint(ops{1},ops{2}.avalue*A); % by derivative of op1
%                     incadjoint(ops{2},A*ops{1}.avalue); % by derivative of op2
%                 case 'mpower'
%                     assert(ops{2}.avarcount == 0,'power needs to be constant');
%                     switch ops{2}.avalue
%                         case 1
%                             % X^1 == X
%                             incadjoint(ops{1},A);
%                         case 2 % FIX ME
%                             X = ops{1}.avalue;
%                             Q = eye(length(X));
%                             incadjoint(ops{1},A*(kron(Q,X)+kron(X',Q)));
%                         case 3 % FIX ME
%                             X = ops{1}.avalue;
%                             incadjoint(ops{1},A*(kron((X')^2,eye(length(X)))+kron(X',X)+kron(eye(length(X)),X^2)));
%                         otherwise
%                             error('not implemented generic power');
%                     end
%                 case 'power'
%                     assert(ops{2}.avarcount == 0,'power needs to be constant');
%                     switch ops{2}.avalue
%                         case 1
%                             % X^1 == X
%                             incadjoint(ops{1},A);
%                         case 2
%                             % same as X*X = X*A+A*X
%                             incadjoint(ops{1},2*A*ops{1}.avalue);
%                         case -1
%                             incadjoint(ops{1},-V*A*V);                            
%                         otherwise
%                             incadjoint(ops{1},ops{2}.avalue*diag(ops{1}.avalue.^(ops{2}.avalue-1)));
%                     end
%                 case 'logdet'
%                     q = inv(V)';
%                     incadjoint(ops{1},q(:)');
%                 case 'det'
%                     assert('Not implemented autodiff of det');
%                 case 'trace'  
%                     % ac.uk says: eye(n)(:)'
%                     % not totally correct
%                     
%                     incadjoint(ops{1},A); %eye(length(ops{1}.avalue))*A);
%                 case 'vec' % vectorization
%                 case 'inv' % inversion
%                     incadjoint(ops{1},-V*A*V);
%                 case 'transpose'
%                     incadjoint(ops{1},A');
%                 case ''  % nothing
%                     return
%                 otherwise
%                     error(['Unimplemented ' this.aop]);
%             end
%             this.aoperands = ops;
%             
%             % then continue the descent ONLY if meaningful
%             for I=1:length(this.aoperands)
%                 if this.aoperands{I}.avarcount > 0
%                     sautodiff(this.aoperands{I});
%                 end
%             end
%         end
        
        
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
       
        
        function this = incadjoint(this,value)
            this.aadjoint = this.aadjoint + value;
            
        end
        
        % sets the value for the constant or variable ones
        function set(this,value)
            assert(isempty(this.aop));
            this.avalue = value;
        end         
    end
    
end

