import numpy as np
import math

def embed_matrix(A: np.ndarray):
    """
    Embed a dxd matrix A into a 2^n x 2^n matrix, where n = ceil(log2(d)).

    Args:
        A (np.ndarray): Matrix to embed.
    Returns:
        A_embedded (np.ndarray): 2^n x 2^n matrix with embedded matrix A.
    """
    d = A.shape[0]
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    n = math.ceil(math.log2(d))
    D = 2 ** n
    A_embedded = np.eye(D, dtype=A.dtype)
    A_embedded[:d, :d] = A

    return A_embedded, n
    
def generate_binary_matrix(length):
    n = 2 ** length
    nums = np.arange(n, dtype=np.uint32)
    return ((nums[:, None] & (1 << np.arange(length-1, -1, -1))) > 0).astype(int)
        
def compute_cq_and_unitary(bits=None, delta=0.001, L=2):
    """
    Infers the quantum memory states [PRA 101(3), 032327 (2020)], merges quantum memory states with
    the equivalence relation, then builds the unitary operator [PRL 120, 240502 (2018)] for
    implementation on a quantum computer.

    The unitary operator is then applied in a quantum circuit and only the bottom-most qubit is measured.
    E.g. for a 4-qubit unitary:
        0: -U---
        1: -U---
        2: -U---
        3: -U-M-

    Example of an implementation:
    ```python
    from qibo import Circuit, gates
    circ = Circuit(4)
    circ.add(gates.Unitary(Unitary0, 1, 2, 3))
    circ.add(gates.M(3))
    result = circ(nshots=int(1e4)).frequencies()
    ```
    
    Arguments:
        bits (np.ndarray): 1D numpy array of {0, 1} binary time series.
        delta (float): Tolerance for state merging. Defaults to 0.001. Delta is automatically
            increased if non-unifilar transitions are detected.
        L (int): Length of past for inference.
    Returns:
        Cq (float): Quantum statistical memory.
        Unitary (np.ndarray): 2D numpy array representing the unitary operator for the quantum
            model.
        QMS (np.ndarray): 2D numpy array of the quantum memory states.
    """
    
    
    if bits is None:
        return np.nan

    while True:
        # Construct a matrix with increasing timesteps
        windows = np.lib.stride_tricks.sliding_window_view(bits, window_shape = L+1)
        
        sub = windows[:, :L]
        k = sub.shape[1]
        weights = 2 ** np.arange(k-1, -1, -1)
        ints = sub @ weights
        
        # Find conditional probabilities
        condCounts = np.zeros((2**L, 2))
        for i in range(0, np.shape(windows)[0]):
            row = ints[i]
            col = windows[i][L]
            condCounts[row][col] += 1
        
        row_sums = condCounts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        condProbs = condCounts / row_sums
        
        past = generate_binary_matrix(L)
        future = past.copy()
        
        # Evaluating conditional probabilities -- contatenated
        condProbs_concatenated = np.zeros((2**L, 2**L))
        for ii in range(2**L):
            for jj in range(2**L):
                
                # Step 1
                temp = np.hstack( [past[ii, :], future[jj, :]] )
        
                # Step 2
                mat = np.zeros((L, L+1))
                for kk in range(L):
                    mat[kk, :] = temp[kk : kk+L+1]
        
                # Step 3
                prob_from_mat = []
                for kk in range(L):
                    row_to_access = int(mat[kk, 0:L].dot(2 ** np.arange(mat[kk, 0:L].size-1, -1, -1)))
                    col_to_access = int(mat[kk, L])
                    prob_from_mat.append(float(condProbs[row_to_access, col_to_access]))
        
                condProbs_concatenated[ii, jj] = np.prod(prob_from_mat)
        
        # Flip to colunn-wise states (each col sums to 1)
        condProbs_concatenated = condProbs_concatenated.T
        
        # Quantum memory states (each col contains probability amplitudes, colwise-elements**2 sums to 1)
        probamps = np.sqrt(condProbs_concatenated)
        
        # Building / merging states
        sigma_states = np.zeros((np.shape(condProbs_concatenated)[0]))
        for ii in range(np.shape(sigma_states)[0]):
            if np.sum(condProbs_concatenated[ii, :]) > 0:
                sigma_states[ii] = np.max(sigma_states) + 10
        
        indices = []
        for ii in range(np.shape(sigma_states)[0]):
            if np.sum(condProbs_concatenated[ii, :]) > 0:
                indices.append(ii)
        
        # Merge states using inner products <sigma_i | sigma_j>
        # If <o|o> >= 1-delta, same state, else different state
        for ii in range(np.shape(indices)[0]-1):
            for jj in range(ii+1, np.shape(indices)[0]):
                if np.dot(probamps[:, indices[ii]], probamps[:, indices[jj]]) >= 1-delta:
                    # print(f'ii = %d, jj = %d, <o|o> = %.4f' %(indices[ii], indices[jj], probamps[:, indices[ii]].T @ probamps[:, indices[jj]]))
                    sigma_states[indices[jj]] = sigma_states[indices[ii]]
        
        # Reducing to consecutive state numbers
        stateno = np.zeros((np.shape(sigma_states)[0]))
        for ii in range(1, int(max(sigma_states)/10)+1):
            x = np.where(sigma_states == ii * 10)[0]
            if len(x) == 1:
                stateno[x[0]] = max(stateno) + 1
            elif len(x) > 1:
                temp = int(np.max(stateno)) + 1
                for jj in range(len(x)):
                    stateno[x[jj]] = temp
        
        sigma_states = stateno.copy()
        stateno = np.repeat(sigma_states[:, None], 2**L, axis=1)
        
        windows2 = np.lib.stride_tricks.sliding_window_view(bits, window_shape = 2*L)
        state_trans = np.zeros(np.shape(windows2)[0])
        
        for ii in range(np.shape(windows2)[0]):
            idx_r = windows2[ii, :L].dot(2 ** np.arange(windows2[ii, :L].size - 1, -1, -1))
            idx_c = windows2[ii, L:].dot(2 ** np.arange(windows2[ii, L:].size - 1, -1, -1))
            state_trans[ii] = stateno[idx_r, idx_c]
        
        # Build transition matrix: rows represent pasts, cols represent futures
        # Note: If state each past state transits to >2 other future states, it implies non-unifilarity.
        # If non-unfilarity occurs, adjust delta s.t. each past state transits to <= 2 future states.
        Tcount = np.zeros((int(np.max(stateno)), int(np.max(stateno))))
        for ii in range(len(state_trans)-1):
            row = int(state_trans[ii]-1)   # past state
            col = int(state_trans[ii+1]-1) # future state
            Tcount[row, col] += 1
        
        row_sums = Tcount.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        Tprob = Tcount / row_sums
        
        # Check columns of Tprob for >2 entries per column. If true, repeat the above but increase delta.
        # If state transits to >2 other states, it implies non-unfilarity. Adjust delta until each state transits to <=2 other states.
        entry_count = np.zeros((np.shape(Tprob)[1], ))
        for jj in range(np.shape(Tprob)[1]):
            for ii in range(np.shape(Tprob)[0]):
                if Tprob[ii, jj] != 0:
                    entry_count[jj] += 1
        if np.max(entry_count) > 2:
            delta += np.sqrt(1/len(bits))
            # print(f"Non-unifilarity detected; current delta tolerance {delta} increased by {np.sqrt(1/len(bits))}.")
        else:
            break
    
    num_causal_states = int(np.max(sigma_states))
    
    # Finding probability of each unmerged state
    windows3 = np.lib.stride_tricks.sliding_window_view(bits, window_shape = L)
    stateProbCount = np.zeros((2**L, 1))
    for ii in range(len(windows3)):
        row = windows3[ii, :].dot(2 ** np.arange(windows3[ii, :].size - 1, -1, -1))
        stateProbCount[row] += 1
    stateProbVec = stateProbCount / np.sum(stateProbCount)
    
    # Sqrt-ed conditional futures for the states
    independent_sigma_states = []
    condProbs_concatenated_independent_sigma = np.zeros((2**L, int(np.max(sigma_states))))
    for ii in range(int(np.max(sigma_states))):
        independent_sigma_states.append(ii)
        vec = np.where(sigma_states == ii+1)[0]
        total_prob_for_that_state = np.sum(stateProbVec[vec])
        if total_prob_for_that_state == 0:
            continue
    
        for jj in range(len(vec)):
            temp = (stateProbVec[vec[jj]] / total_prob_for_that_state) * condProbs_concatenated[:, vec[jj]]
            condProbs_concatenated_independent_sigma[:, ii] += temp
    
    probamps_independent_sigma = np.sqrt(condProbs_concatenated_independent_sigma)
    
    # Finding probability of each state
    windows4 = np.lib.stride_tricks.sliding_window_view(bits, window_shape = L) # also == windows3
    stateProbCount = np.zeros((2**L, 1))
    for ii in range(len(windows4)):
        row = windows4[ii, :].dot(2 ** np.arange(windows4[ii, :].size - 1, -1, -1))
        stateProbCount[row] += 1
    stateProbVec = stateProbCount / np.sum(stateProbCount)
    prob_of_independent_sigma = np.zeros(int(np.max(sigma_states)))
    for ii in range(int(np.max(sigma_states))):
        vec2 = np.where(sigma_states == ii+1)[0]
        vec3 = []
        for jj in range(len(vec2)):
            coltoextract = vec2[jj]
            vec3.append(stateProbVec[coltoextract])
        prob_of_independent_sigma[ii] = np.sum(vec3)
    
    # Density matrix of merged states
    rho_merged_states = np.zeros((2**L, 2**L))
    for ii in range(int(np.max(sigma_states))):
        rho_merged_states += prob_of_independent_sigma[ii] * np.outer(probamps_independent_sigma[:, ii], probamps_independent_sigma[:, ii])
    
    # Calculating Cq
    eigvals = np.linalg.eigh(rho_merged_states)[0]
    eigvals = np.real(eigvals)
    eigvals = eigvals[eigvals > 0]
    Cq = -np.sum(eigvals * np.log2(eigvals))
    
    # Gram Schmidt
    VVV = probamps_independent_sigma.copy()
    n, k = VVV.shape
    
    UUU = np.zeros((n, k))
    UUU[:, 0] = VVV[:, 0] / float(np.sqrt(np.dot(VVV[:, 0], VVV[:, 0])))
    
    for ii in range(1, k):
        UUU[:, ii] = VVV[:, ii]
        for jj in range(0, ii):
            to_minus = np.dot(UUU[:, ii], UUU[:, jj]) / np.dot(UUU[:, jj], UUU[:, jj]) * UUU[:, jj]
            for idx, kk in enumerate(to_minus): # Check for NaN, resulting from division by zero. If NaN, set to zero.
                if np.isnan(kk):
                    to_minus[idx] = 0
            UUU[:, ii] = UUU[:, ii] - (to_minus)
        UUU[:, ii] = UUU[:, ii] / np.sqrt( np.dot(UUU[:, ii], UUU[:, ii]) )
    ortho_basis_sigma = UUU
    
    # Cij matrix
    # Cij = np.zeros((k, k))
    # for ii in range(k):
    #     for jj in range(k):
    #         Cij[ii, jj] = np.dot( probamps_independent_sigma[:, ii], probamps_independent_sigma[:, jj] ) 
    Cij = probamps_independent_sigma.T @ probamps_independent_sigma
    
    # Finding coefficients using \ or modivide (in Matlab)
    coefficients = np.zeros(( np.shape(probamps_independent_sigma)[1], k))
    for ii in range(k):
        LHS = probamps_independent_sigma[:, :ii+1]
        RHS = ortho_basis_sigma[:, ii]
        
        sol, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
        forwardslash_vec = sol
        coefficients[ii, :len(sol)] = sol
    
    condprobs_singletimestep = np.zeros((2, int(np.max(sigma_states))))
    for jj in range(int(np.max(sigma_states))):
        condprobs_singletimestep[0, jj] = np.sum(condProbs_concatenated_independent_sigma[:2**L // 2, jj])
        condprobs_singletimestep[1, jj] = np.sum(condProbs_concatenated_independent_sigma[2**L // 2 : 2**L, jj])
    
    # Computing Cq, not merged (skipped)
    
    # Getting nextstate_singletimestep
    transition_matrix = Tprob.T
    
    probamp_singletimestep = np.sqrt(condprobs_singletimestep)
    
    # Assign temp variables
    temp_transition_matrix = transition_matrix.copy()
    for ii in range(np.shape(temp_transition_matrix)[0]):
        for jj in range(np.shape(temp_transition_matrix)[1]):
            if temp_transition_matrix[ii, jj] == 0:
                temp_transition_matrix[ii, jj] = np.nan # To prevent np.around(value, 0) = 0 from coinciding with other exact 0 values. This affects finding nextstate_singletimestep.
    
    temp_condfut_singletimestep = condprobs_singletimestep
    # print('temp_condfut_singletimestep =\n', temp_condfut_singletimestep)
    nextstate_singletimestep = np.zeros(( np.shape(temp_condfut_singletimestep)[0], np.shape(temp_condfut_singletimestep)[1] ))
    # Go through each column. If each column fails, adjust rounding_dp
    for jj in range(np.shape(temp_condfut_singletimestep)[1]):
        rounding_DP = 15
        
        while True:
            nextstate_singletimestep[:, jj] = 0
            for ii in range(np.shape(temp_condfut_singletimestep)[0]):
                if temp_condfut_singletimestep[ii, jj] != 0:
        
                    at_row_of_transition_matrix = np.where( np.around(temp_transition_matrix[:, jj], rounding_DP) == np.around(temp_condfut_singletimestep[ii, jj], rounding_DP) )
                    if len(at_row_of_transition_matrix[0]) == 0:
                        nextstate_singletimestep[ii, jj] = np.nan
        
                    elif len(at_row_of_transition_matrix[0]) == 2:
                        if temp_transition_matrix[int(at_row_of_transition_matrix[0][0]), jj] == temp_transition_matrix(int(at_row_of_transition_matrix[0][1]), jj):
                            nextstate_singletimestep[ii, jj] = at_row_of_transition_matrix[0][ii]
        
                    else:
                        nextstate_singletimestep[ii, jj] = at_row_of_transition_matrix[0][0]
                # print('nextstate_singletimestep =\n', nextstate_singletimestep)
            if np.any(np.isnan(nextstate_singletimestep[:, jj])):
                rounding_DP -= 1
                # if rounding_DP < 0:
                    # raise RuntimeError(f"Rounding failed to resolve column {jj}")
                continue # Retry with lower precision
            else:
                break # Success
    
    nextstate_singletimestep += 1
    
    num_sigma = int(np.max(sigma_states))
    Unitary_nonzeros = np.zeros((num_sigma * 2, num_sigma))
    for jj in range(num_sigma):   # MATLAB: 1:max(sigma_states)
        Unitary_nonzeros_row = 0
        for ii in range(num_sigma):
            for output in range(2):   # MATLAB: 0:1
                output_idx = output + 1   # MATLAB indexing logic
                Unitary_nonzeros_row += 1
                Unitary_nonzeros_col = jj
                inner_matrix = np.zeros((ii + 1, jj + 1)) # MATLAB: zeros(ii, jj)
                for i in range(ii + 1):
                    for j in range(jj + 1):
    
                        # coefficient for idx_i
                        row_coeff_idx_i = ii
                        T1 = coefficients[row_coeff_idx_i, i]
    
                        # coefficient for idx_i'
                        row_coeff_idx_iprime = jj
                        T2 = coefficients[row_coeff_idx_iprime, j]
    
                        # prob_amp stuff
                        row_prob_amp = output_idx - 1   # convert to Python index
                        col_prob_amp = j
                        T3 = probamp_singletimestep[row_prob_amp, col_prob_amp]
                        
                        # Cij stuff
                        row_Cij = i
                        col_Cij = int(nextstate_singletimestep[row_prob_amp, j])
    
                        if col_Cij == 0:
                            T4 = 0
                        else:
                            T4 = Cij[row_Cij, col_Cij-1]
                            
                        inner_matrix[i, j] = T1 * T2 * T3 * T4
    
                Unitary_element = inner_matrix.sum()
                Unitary_nonzeros[Unitary_nonzeros_row - 1, Unitary_nonzeros_col] = Unitary_element
    
    # Normalizing the Unitary_nonzeros
    # for j in range(Unitary_nonzeros.shape[1]):
    #     n = np.linalg.norm(Unitary_nonzeros[:, j])
    #     if n != 0:
    #         Unitary_nonzeros[:, j] /= n
    norms = np.linalg.norm(Unitary_nonzeros, axis=0)
    nonzero = norms != 0
    Unitary_nonzeros[:, nonzero] /= norms[nonzero]
    
    # Assign rand numbers to zero-cols
    # Then do a Gram Schmidt to orthonormalize them
    n = Unitary_nonzeros.shape[0]
    Unitary_notyetreorder = np.zeros((n, n))
    num_cols = Unitary_nonzeros.shape[1]
    Unitary_notyetreorder[:, :num_cols] = Unitary_nonzeros
    startingcol = num_cols   # Python index (already shifted)
    
    # filling with rand
    for j in range(startingcol, n):
        Unitary_notyetreorder[:, j] = np.random.rand(n)
    
    # Gram Schmidt the remaining zero-cols
    nn, kk = Unitary_notyetreorder.shape
    U_temp = np.zeros((nn, kk))
    U_temp[:, :startingcol] = Unitary_nonzeros
    
    for i in range(startingcol, kk):
        U_temp[:, i] = Unitary_notyetreorder[:, i].copy()
        for j in range(i):
            proj = (U_temp[:, i].T @ U_temp[:, j]) / (U_temp[:, j].T @ U_temp[:, j])
            U_temp[:, i] = U_temp[:, i] - proj * U_temp[:, j]
        # Normalize
        U_temp[:, i] = U_temp[:, i] / np.sqrt(U_temp[:, i].T @ U_temp[:, i])
        Unitary_notyetreorder[:, i] = U_temp[:, i]
    
    # Reorder the cols of Unitary
    Unitary_zerocols = Unitary_notyetreorder[:, startingcol:]
    n = Unitary_nonzeros.shape[0]
    Unitary = np.zeros((n, n))
    num_cols = Unitary_nonzeros.shape[1]
    for i in range(num_cols):
        Unitary[:, 2*i] = Unitary_nonzeros[:, i]
        Unitary[:, 2*i + 1] = Unitary_zerocols[:, i]
    
    # Generating the merged quantum memory states
    num_sigma = int(np.max(sigma_states))
    QMS = np.zeros((num_sigma, num_sigma))
    QMS[0, 0] = 1
    for j in range(1, num_sigma):
        QMS[0, j] = Cij[0, j]
    if QMS.shape[0] > 1:
        QMS[1, 1] = np.sqrt(1 - QMS[0, 1]**2)
        for j in range(2, num_sigma):
            for i in range(1, j):
                vec = np.zeros(i + 1)
                vec[0] = Cij[i, j]
                for ii in range(1, i + 1):
                    vec[ii] = -QMS[ii - 1, i] * QMS[ii - 1, j]
                Denominator = QMS[i, i]
                QMS[i, j] = vec.sum() / Denominator
            vecc = QMS[:j, j]
            vecc_sq = vecc**2
            vecc_sq = -vecc_sq
            QMS[j, j] = np.sqrt(1 + vecc_sq.sum())

    # Embed Unitary if not 2^n x 2^n for n = 1, 2, 3, ...
    Unitary, _ = embed_matrix(Unitary)
    
    return Cq, Unitary, QMS