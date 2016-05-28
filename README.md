# Matlab Reverse-Mode Matrix-aware Second Order Automatic Differentiation 
Emanuele Ruffaldi 2016, Scuola Superiore Sant'Anna

This small package provides a mechanism for performing First and Second order differentiation of Matrix-Matrix functions using the Reverse AutoDifferentiation paradigm. The first order is relatively simple and does not use tensors.

The second order modality provides the Hessan Matrix (Tensor) based on the edge pushing algorithm. This algorithm (Gower et al 2010,2012) is known to be efficient but limited to scalar (or scalarized) functions. In this work we extend it to matrix form allowing for efficient matrix hessians.

# Usage 

Here a brief example of first order

    % Declare some variables. The actual content can be replaced any time, and it can be even symbolics from Symbolic Toolbox
    k1 = 3;
    k2 = 4;
    b = matexp('b',eye(k1));
    q = matexp('q',rand(k1,k2);

    f = trace(b.^3)*ones(k1,k2)+q;
    update(f); % updates the function
    value(f); % the value of f
    autodiff(f); % computes the first order
    adjoint(b); % the J(f,b) flattened 
    adjoint(q); % the J(f,q) flattened

The second order uses another approach and it can be used to compute first and second order at once (so no need to call autodiff)

# Note about tensors

Tensors (multidimensional array) are useful for representing jacobians of matrix-matrix functions. A generic matrix function F depends
on a series of variable Xi with i=1..k of dimensions [mi,ni] and produces an output [mf,nf]. This means that each partial derivative
J(F,Xi) is a tensor [m_f,n_f,m_i,n_i]. 

In practice working with tensors is not easy, in particular for the generalized product (not available natively in Matlab), and not
compatible with symbolic toolbox when we want to do a symbolic verification of the result.

For this reason we follow the approach of flattening of the matrices using the column notation: X: that is a column vector of dimension [mnX,1].
This means that the Jacobian becomes a matrix [mn_f,mn_i] This speeds up a lot the computations.

For the hessian matrix the situation is similar. For every pair of matrix variable (Xi,Xj) the Hessian Matrix is a tensor as follows:
[m_f,n_f,m_i,n_i,m_j,n_j]. Note that the Hessian Tensor is symmetrix for each F output, that is 
the matrix [A,B,:,:,:,:] is symmetric. The Hessian Tensor can be flattened in several ways: if we consider the equivalent computation using 
the symbolic toolbox we can have one example of flattening:
    
    f = ... [m_f,n_f]
    vars = ... vector of variables each [m_i,n_i] each flattened [mn_i,1]
    J = jacobian(f(:),vars); % [mn_f, sum mn_i]
    H = jacobian(J(:),vars); % [mn_f*sum mn_i, sum mn_i]

This means that the matrix rows contains all the outcomes of the Jacobian and the columns the second derivatives. For the way
Matlab performs stacking it will outer iterate by rows so we can retrieve the
    
    br = blkidx(3,(sum_mn_i)); % returns the range extrema [s,e] for accessing the 3rd of size sum_mn_i
    Hi = H(br(1):br(2),:); % symetric matrix for the 3rd output of f
    

# Limitations

First order: many cases have been tested but this is not production ready tool

Second order: there are several operations not yet supported, and also we rely on tensors for the internal manipulation meaning that
we cannot use symbolic toolbox verification, and we rely on the tprod function provided on file exchange and update in this github: 
https://github.com/eruffaldi/tprod.





# Implementation

The implementation is based on the handle-derived class matexp that allows to express a matlab matrix expression using operator overloading.
The matexp is then manipulated for supporting the derivation in recursive mode.

The Hessian Matrix is computed using edge push algorithm that streamline and simplifies the process.

# Tests

First order is tested in multi_test that takes many expressions and compare them against symbolic toolbox

# References


Test functions: http://camo.ici.ro/journal/vol10/v10a10.pdf
Matrix Calculus Functions: http://www.ee.ic.ac.uk/hp/staff/dmb/matrix/calculus.html

Edge-push papers
1) Gower, R. M., and M. P. Mello. "A new framework for the computation of Hessians." Optimization Methods and Software 27.2 (2012): 251-273.
2) (public) Gower, Robert Mansel, and Margarida P. Mello. Hessian matrices via automatic differentiation. Universidade Estadual de Campinas, Instituto de Matem?tica, Estat?stica e Computa??o Cient?fica, 2010.
