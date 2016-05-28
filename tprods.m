% einstein summation product of tensor
%
% r = tprods(A,iA,B,iB)
%
% iA and iB contains as negative indices the corresonding dimensions. There
% is no need to be contiguous just some negative markers
%
% For numeric tensors use tprod
%
% TO BE COMPLETED
%
% NOTE: in any case it will not work with symbolics
function r = tprods(A,iA,B,iB)

iAn = find(iA < 0);
iBn = find(iB < 0);


sA = size(A);
sB = size(B);
[soiAn,isoiAn] = unique(iA(iAn));
[soiBn,isoiBn] = unique(iB(iBn));
assert( soiAn == length(iAn),'no duplicates');
assert( soiBn == length(iBn),'no duplicates');
assert( all(soiAn) == all(soiBn),'same dimensions');
assert(length(iAn) == length(iBn),'same summations');
assert(all(sA(isoiAn),sB(isoiBn)),'same corresponding dimensions');

