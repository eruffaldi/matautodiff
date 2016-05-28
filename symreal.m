% helper function for testing, returns a real symbolic matrix of size
function r = symreal(name,size)
r = sym(sym(name,size),'real');
