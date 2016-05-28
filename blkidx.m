% given blocksize returns indices for accessing the blocks via index
% returns [s,e] for each row for i=s:e
function br = blkidx(indices,blocksize)

br1 = (indices(:)-1)*blocksize+1;
br2 = br1+blocksize-1;
br = [br1,br2];