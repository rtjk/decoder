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

def fixed_counters_decrement(Htr_sparse, H_sparse, r, v):
    fixed_decr = []
    for block in range(2):
        #Set up syndrome for current block
        dense_s_block = vector(ZZ,r)
        for i in range(v):
            pos_toggle = Htr_sparse[block][i]
            dense_s_block[pos_toggle] = 1
        # alla fine di questo abbiamo densificato le colonne
        # ottengo il decremento accumulando lo shift di HTr[0] per le H_sparse
        decr = compute_upcs(H_sparse, dense_s_block, r, v)
        fixed_decr.append(decr)
    return fixed_decr

################################################################################

def counters_update_steroids(Htr_sparse, H_sparse, fixed_decr, s, r, v, pos, upc):

    block = pos//r
    shift = pos%r
    
    shifted_incr = vector(ZZ, 2*r)
    for i in range(2):
        for j in range(r):
            new_pos = i*r + ((j+shift)%r)
            shifted_incr[new_pos] = fixed_decr[block][i*r+j]
    
    upc += (-shifted_incr)
    
    #Now, correct 
    for i in range(v):
        row_index = (Htr_sparse[block][i] + shift)%r
        d = s[row_index] #if d = 1, decrease counters; otherwise, increase
        
        if d == 0:
            #update all counters
            for j in range(2):
                for ell in range(v):
                    pos_toggle = j*r + (H_sparse[j][ell] + row_index)%r
                    upc[pos_toggle] += 2

    return upc

################################################################################

def bfmax_steroids(Htr_sparse, H_sparse, fixed_decr, s, r, v, num_iter_max):
    
    dec_e = vector(GF(2), 2*r) #error estimate
    
    w_s = s.list().count(1) #syndrome weight
    num_iter = 0
    
    upc = compute_upcs(H_sparse, s.change_ring(ZZ), r, v)
    
    while (w_s != 0) & (num_iter < num_iter_max):
        
        pos = argmax(upc)

        print(f"At iteration {num_iter} the value of the max counter is {upc[pos]}")

        #Flip position in dec_e
        dec_e[pos] += 1
        
        #Update counters
        upc = counters_update_steroids(Htr_sparse, H_sparse, fixed_decr, s, r, v, pos, upc);

        #Flip syndrome
        block = pos // r
        shift = pos % r
        for i in range(v):
            pos_toggle = (Htr_sparse[block][i]+shift)%r
            s[pos_toggle] += 1
        
        #Update syndrome weight and number of iterations
        w_s = s.list().count(1) #syndrome weight        
        num_iter += 1
        
    return dec_e

################################################################################

fixed_decr = fixed_counters_decrement(Htr_sparse, H_sparse, r, v)

################################################################################
