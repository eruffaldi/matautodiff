% scratchpad for some J2 tests
a = symreal('a',[3,4]);

f = a.^3;
J = jacobian(f(:),a(:));
H = jacobian(J(:),a(:));

