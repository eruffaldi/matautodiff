Test Cases for Hessian:
- complete sinquad

For Diff
- add special operation as rotation matrix (e.g. for kinematics) and quaternion operations

For Hessian
- .^
- ^
- remove need for tensor operations => allowing for symbolic testing

Optimize
- diag
- X^k k > 2
		https://books.google.it/books?id=N871f_bp810C&pg=PA353&lpg=PA353&dq=hessian+chain+rule&source=bl&ots=st28mYrWrY&sig=ZNuWzqFIIVmf6Rc6INBh3dmRIU8&hl=en&sa=X&ved=0ahUKEwiA25nOv-HMAhXBJ5oKHdsxA1AQ6AEIITAC#v=onepage&q=hessian%20chain%20rule&f=false
- vec(ab') = b kron a
- vech(A sym) = vec(A) | sym
- Kmn vec(A) = vec('A..)