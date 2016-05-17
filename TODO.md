For Diff
- add special operation as rotation matrix (e.g. for kinematics) and quaternion operations

For Hessian
- .*
- * (should use D kron)
- scalar function
- kron (will not implement)
- power
- mpower
- det
- logdet
- trace
- transpose
- diag
- vec

Optimize
- diag
- X^k k > 2
		https://books.google.it/books?id=N871f_bp810C&pg=PA353&lpg=PA353&dq=hessian+chain+rule&source=bl&ots=st28mYrWrY&sig=ZNuWzqFIIVmf6Rc6INBh3dmRIU8&hl=en&sa=X&ved=0ahUKEwiA25nOv-HMAhXBJ5oKHdsxA1AQ6AEIITAC#v=onepage&q=hessian%20chain%20rule&f=false
- vec(ab') = b kron a
- vech(A sym) = vec(A) | sym
- Kmn vec(A) = vec('A..)