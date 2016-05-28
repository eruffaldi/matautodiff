function br = blkidx(indices,blocksize)

br1 = (indices(:)-1)*blocksize+1;
br2 = br1+blocksize-1;
br = [br1,br2];