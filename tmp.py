# r = P
# v = V
# s = syndrome

################################################################################

def compute_upcs(H_sparse, s, r, v):
    upc = vector(ZZ,2*r)
    for i in range(2):
        for j in range(v):
            shift_val = H_sparse[i][j]
            #shift syndrome
            delta_upc = vector(ZZ,r)
            for k in range(r):
                new_pos = (k+shift_val)%r;
                delta_upc[new_pos] = s[k]
            upc[i*r:(i+1)*r] += delta_upc
    return upc

################################################################################

def compute_fixed_decr(Htr_sparse, H_sparse, r, v):
    fixed_decr = []
    for block in range(2):
        dense_s_block = vector(ZZ,r)
        # copy first column of the block
        for i in range(v):
            pos_toggle = Htr_sparse[block][i]
            dense_s_block[pos_toggle] = 1
        decr = compute_upcs(H_sparse, dense_s_block, r, v)
        fixed_decr.append(decr)
    return fixed_decr

################################################################################

def update_upc(Htr_sparse, H_sparse, fixed_decr, s, r, v, pos, upc):
    block = pos//r
    shift = pos%r
    # update upcs: -1
    shifted_decr = vector(ZZ, 2*r)
    for i in range(2):
        for j in range(r):
            new_pos = i*r + ((j+shift)%r)
            shifted_decr[new_pos] = fixed_decr[block][i*r+j]
    upc += (-shifted_decr)
    # update upcs: +2
    for i in range(v): # scan each idx of col corresponding to flipped pos
        row_index = (Htr_sparse[block][i] + shift)%r
        if s[row_index] == 0: # if syndrome bit is 0
            for j in range(2): # scan each index in the row
                for k in range(v):
                    pos_toggle = j*r + (H_sparse[j][k] + row_index)%r
                    upc[pos_toggle] += 2
    return upc

################################################################################

def bfmax_steroids(Htr_sparse, H_sparse, fixed_decr, s, r, v, num_iter_max):
    error = vector(GF(2), 2*r)
    hw = s.list().count(1)
    num_iter = 0
    upc = compute_upcs(H_sparse, s.change_ring(ZZ), r, v)
    while (hw != 0) & (num_iter < num_iter_max):
        pos = argmax(upc)
        error[pos] += 1
        upc = update_upc(Htr_sparse, H_sparse, fixed_decr, s, r, v, pos, upc);
        # update syndrome
        block = pos//r
        shift = pos%r
        for i in range(v):
            pos_toggle = (Htr_sparse[block][i]+shift)%r
            s[pos_toggle] += 1
        hw = s.list().count(1)
        num_iter += 1
    return error

################################################################################

fixed_decr = compute_fixed_decr(Htr_sparse, H_sparse, r, v)

################################################################################
