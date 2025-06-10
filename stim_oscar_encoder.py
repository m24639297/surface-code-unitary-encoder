import itertools, stim
from typing import Optional, Tuple, List, Optional
import numpy as np
import tools as sc

r""" Surface code layout & gate scheduling

(XL)               X               X
 j
 ↑     + ───── + ───── 0 ───── + ───── 0   -
 |     │1     2│1      │1     2│1      │    |
 |  Z  │   X   │   Z   │   X   │   Z   │    | a=1
 |     │       │2      │       │2      │    |
 |     0 ───── + ===== + ===== + ───── 0   -
 |     │1      ǁ 1   2 ǁ 1     ǁ1     2│1
 |     │   Z   ǁ   X   ǁ   Z   ǁ   X   │  Z
 |     │2      ǁ       ǁ 2     ǁ       │2
 |     + ───── + ===== + ===== 0 ───── +   
 |     │1     2ǁ 1     ǁ 1   2 ǁ1      │
 |  Z  │   X   ǁ   Z   ǁ   X   ǁ   Z   │ 
 |     │       ǁ 2     ǁ       ǁ2      │
 |     0 ───── o ===== 0 ===== 0 ───── 0   
 |     │1      │1     2│1      │1     2│1
 |     │   Z   │   X   │   Z   │   X   │  Z
 |     │2      │       │2      │       │2
 |     0 ───── + ───── 0 ───── + ───── +
 |      1     2         1     2
           X               X
     --------------------------------------> i (ZL)

          
Initial pattern for rot code (in the middle, thick border): 
    bot-left qubit (o): |psi>
    NE-SW diagonal and to its right (0): |0>
    the rest (x): |+>
    Fixed X stab: j > i
    Fixed Z stab: j < i
"""
def get_new_plus_anc_locs(k: int, offset: tuple = None):
    d = 2 * k + 1
    locs =  [(x, -1) for x in range(0, d, 2)]
    locs += [(d, y) for y in range(-1, d, 2)]
    locs += [(x, d) for x in range(0, d, 2)]
    locs += [(-1, y) for y in range(1, d+1, 2)]
    return sc.offset_loc_list(locs, offset)

def get_new_anc_locs(k: int, offset: tuple = None):
    d = 2 * k + 1
    locs =  [(x, y) for y in [-1, d] for x in range(-1, d + 1)]
    locs += [(x, y) for x in [-1, d] for y in range(d)]
    return sc.offset_loc_list(locs, offset)

def expanding_gate_seq(k, offset, flattened: bool=True):
    d = 2 * k + 1
    gate_seq = []
    # adding gates on Bot, Top, Left, Right boundaries
    # 1
    gates = []
    gates += [[(i-1, 0), (i-1, -1)]     for i in range(d)]
    gates += [[(i+1, d-1), (i+1, d)]    for i in range(d)]
    gates += [[(-1, j+1), (0, j+1)]     for j in range(d)]
    gates += [[(d, j-1), (d-1, j-1)]    for j in range(d)]
    gate_seq.append(list(itertools.chain(*gates)))

    # 2
    gates = []
    gates += [[(x, -1), (x-1, -1)]  for x in range(0, d, 2)]
    gates += [[(x, d), (x+1, d)]    for x in range(0, d, 2)]
    gates += [[(-1, y), (-1, y-1)]  for y in range(1, d+1, 2)]
    gates += [[(d, y), (d, y+1)]    for y in range(-1, d-1, 2)]
    gate_seq.append(list(itertools.chain(*gates)))

    # 3
    gates = [(0, 0), (-1, -1), (d, -1), (d-1, 0), (d-1, d-1), (d, d), (-1, d), (0, d-1)]
    gate_seq.append(gates)

    # 4
    gates = []
    gates += [[(x, 0), (x-1, -1)]   for x in range(2, d, 2)]
    gates += [[(x, d-1), (x+1, d)]  for x in range(0, d-1, 2)]
    gates += [[(-1, y), (0, y-1)]   for y in range(1, d, 2)]
    gates += [[(d, y), (d-1, y+1)]  for y in range(1, d, 2)]                

    gate_seq.append(list(itertools.chain(*gates)))

    gate_seq =  list(map(lambda lst: sc.offset_loc_list(lst, offset), gate_seq))
    if flattened:
        return gate_seq
    pair_list = lambda it: list(zip(it, it))
    gate_seq = [pair_list(iter(gates)) for gates in gate_seq]
    return gate_seq

def OscarEncoderCircuit(
        k_final: int, 
        basis: str, 
        p: float, 
        k_init: int = 1, 
        init_pattern: str = 'general', 
        post_select_rounds : Optional[int] = None,
        p_init: float = None,
        p2: float = None,
        pm: float = None,
        idle_per_round: bool = False,
        idle_per_gate: bool = True,
        init_error: bool = True,
        gate_error: bool = True,
        last_data_error: bool = True,
        first_round_optimization: bool = True,
        ) -> Tuple[stim.Circuit, Optional[List[np.uint8]]]:
    
    """
    Args: 
        post_select_round: An optional integer default to None. None means to do
            one round of perfect syndrome measurements at the beginning and fix
            gauge accordingly. An integer value means the rounds of noisy
            measurements for post-selection. For the first round of
            measurements, single-measurement detectors are put on fixed
            stabilizers. For multiple noisy rounds, additional detectors
            detecting consecutive measurement outcomes are used for post
            selection. The final gauge fixing operators are applied according to
            the last round of outcomes.  

    Returns:
        A (circuit, mask) tuple. `circuit` is a stim-Circuit for preparing the
        rotated surface code with Oscar's local encoder; `mask` is a list of
        np.uint8 (bit-packed boolean array for whether post-select each
        detector, in little endian) for the `postselection_mask` arg in
        sinter.Task initializer. `mask` is None if `post_select_rounds` is 0,
        i.e. without post-selection. 

    """
    MODES = {
        'data_in_use',
        'circuit',
    }
    PATTERNS = {
        'xz',
        'general'
    }
    basis = basis.upper()
    assert k_final >= k_init, f'k_final={k_final} must >= k_init={k_init}.'
    assert init_pattern in PATTERNS, f'init_pattern must be one of {PATTERNS}.'
    assert basis in 'XYZ', f'basis must be one of X, Y, Z.'
    if basis == 'Y':
        assert init_pattern == 'general', 'Y basis only applicable for init_pattern `general`.'
    p2 = p if p2 is None else p2
    pm = p if pm is None else pm
    p_init = p if p_init is None else p_init

    loc_to_idx = dict()
    idx_to_loc = list()
    d0 = 2 * k_init + 1
    d_final = 2 * k_final + 1
    do_post_select = post_select_rounds is not None
    post_select_rounds = 1 if not post_select_rounds else post_select_rounds

    idx = 0
    for _j in range(d_final):
        for _i in range(d_final):
            loc = (_i, _j)
            idx_to_loc.append(loc)
            loc_to_idx[loc] = idx 
            idx += 1
    for loc in sc.get_rot_anc_locs(d_final, 1):
        idx_to_loc.append(loc)
        loc_to_idx[loc] = idx
        idx += 1
    del idx

    convert_locs_to_indices = lambda _locs: [loc_to_idx[_loc] for _loc in _locs]
    get_offset = lambda k: (k_final - k, k_final - k)
    offset0 = get_offset(k_init)
    data0 = convert_locs_to_indices(sc.get_rot_data_locs(d0, a=1, offset=offset0))
    x_anc0 = convert_locs_to_indices(sc.get_rot_x_anc_locs(d0, a=1, offset=offset0))  
    anc0 = convert_locs_to_indices(sc.get_rot_anc_locs(d0, a=1, offset=offset0))  
    all_qb0 = convert_locs_to_indices(sc.get_rot_all_locs(d0, a=1, offset=offset0))

    circuit = stim.Circuit()
    for idx, loc in enumerate(idx_to_loc):
        circuit.append("QUBIT_COORDS", idx, loc)

    # data initial pattern
    init_dict = {'X': [], 'Z': [], 'Y': []}
    if init_pattern == 'xz':
        if basis == 'X':
            init_dict['X'] = data0
        else:
            assert basis == 'Z'
            init_dict['Z'] = data0
    elif init_pattern == 'general':
        logical_idx = loc_to_idx[offset0]
        init_dict['X'] = convert_locs_to_indices(sc.get_rot_init_in_x_locs(d0, a=1, offset=offset0))
        init_dict['Z'] = convert_locs_to_indices(sc.get_rot_init_in_z_locs(d0, a=1, offset=offset0))
        init_dict[basis] += [logical_idx]
    circuit.append('H', init_dict['X'])
    circuit.append('H', init_dict['Y'])
    circuit.append('S', init_dict['Y'])
        
    # First few rounds of msmt for post selection and gauge fixing
    p0, p_init0, p20, pm0 = (p, p_init, p2, pm) if do_post_select else (0, 0, 0, 0)   
    untouched_data0 = set(data0)  
    if init_error:
        circuit.append('DEPOLARIZE1', data0, p_init0)
    for _idx in range(post_select_rounds):
        if idle_per_round:
            circuit.append('DEPOLARIZE1', all_qb0, p0)
        if init_error:
            circuit.append('DEPOLARIZE1', anc0, p_init0)
        circuit.append('H', x_anc0)
        for sub_idx, gates in enumerate(sc.get_rot_gate_seq(d0, a=1, loc_to_idx=loc_to_idx, offset=offset0)):
            if _idx == 0 and first_round_optimization:
                gates = sc.remove_trivial_cxs(gates=gates, init_dict=init_dict, untouched_data=untouched_data0)
                untouched_data0 = untouched_data0.difference(set(gates))
                # print(f'After gate seq {sub_idx} w/ {len(gates)//2} gates, untouched: {untouched_data0}')
            circuit.append('CX', gates)
            if gate_error:
                circuit.append('DEPOLARIZE2', gates, p20)
            if idle_per_gate:
                idle_qubits = [q for q in all_qb0 if q not in set(gates)]
                circuit.append('DEPOLARIZE1', idle_qubits, p0)
            circuit.append('TICK')

        for loc in sc.get_rot_x_anc_locs(d0, a=1, offset=offset0):
            anc_idx = loc_to_idx[loc]
            circuit.append('H', anc_idx)
            circuit.append('DEPOLARIZE1', anc_idx, pm0)
            circuit.append('MR', anc_idx)
            is_fixed = (init_pattern == 'xz' and basis == 'X') or (init_pattern == 'general' and loc[1] > loc[0])
            if _idx == 0 and is_fixed:
                circuit.append('DETECTOR', [stim.target_rec(-1)], (*loc, 0))
            if _idx > 0:
                circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-d0*d0)], (*loc, 0))
            if _idx == post_select_rounds - 1:
                gauge_locs = sc.get_z_gauge_locs(d0, a=1, loc=loc, offset=offset0)
                gate_args = list(itertools.chain(*[(stim.target_rec(-1), _idx) for _idx in convert_locs_to_indices(gauge_locs)]))
                circuit.append('CZ', gate_args)
 
        for loc in sc.get_rot_z_anc_locs(d0, a=1, offset=offset0):
            anc_idx = loc_to_idx[loc]
            circuit.append('DEPOLARIZE1', anc_idx, pm0)
            circuit.append('MR', anc_idx)
            is_fixed = (init_pattern == 'xz' and basis == 'Z') or (init_pattern == 'general' and loc[0] > loc[1])
            if _idx == 0 and is_fixed:
                circuit.append('DETECTOR', [stim.target_rec(-1)], (*loc, 0))
            if _idx > 0:
                circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-d0*d0)], (*loc, 0))
            if _idx == post_select_rounds - 1:
                gauge_locs = sc.get_x_gauge_locs(d0, a=1, loc=loc, offset=offset0)
                gate_args = list(itertools.chain(*[(stim.target_rec(-1), _idx) for _idx in convert_locs_to_indices(gauge_locs)]))
                circuit.append('CX', gate_args)

        circuit.append('SHIFT_COORDS', [], (0,0,1))
        circuit.append('TICK')

    # Expanding
    for k in range(k_init, k_final):
        offset = get_offset(k)
        d = 2 * k + 1
        fresh_qubits = convert_locs_to_indices(get_new_anc_locs(k, offset))
        all_qubits_this_round = convert_locs_to_indices(sc.get_rot_data_locs(d, 1, offset)) + fresh_qubits
        if idle_per_round:
            circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_data_locs(d+2, 1, get_offset(k+1))), p)
            circuit.append('TICK')
        if init_error:
            circuit.append('DEPOLARIZE2', fresh_qubits, p_init)
        circuit.append('H', convert_locs_to_indices(get_new_plus_anc_locs(k, offset)))
        for gates in expanding_gate_seq(k, offset=offset):
            gate_idxs = convert_locs_to_indices(gates)
            circuit.append('CX', gate_idxs)
            if gate_error:
                circuit.append('DEPOLARIZE2', gate_idxs, p2)
            if idle_per_gate:
                idle_qubits = [q for q in all_qubits_this_round if q not in set(gate_idxs)]
                circuit.append('DEPOLARIZE1', idle_qubits, p)
            circuit.append('TICK')
            
    # data qubit error on final code
    if last_data_error: 
        circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_data_locs(d_final, a=1)), p)

    # one round of perfect stabilizer measurements
    x_anc_indices = convert_locs_to_indices(sc.get_rot_x_anc_locs(d_final, 1))
    anc_indices = convert_locs_to_indices(sc.get_rot_anc_locs(d_final, 1))
    circuit.append('R', anc_indices)
    circuit.append('H', x_anc_indices)
    for gates in sc.get_rot_gate_seq(d_final, 1, loc_to_idx):
        circuit.append('CX', gates)
        circuit.append('TICK')
    circuit.append('H', x_anc_indices)
    circuit.append('MR', anc_indices)
    num_anc = len(anc_indices)
    for k in range(num_anc):
        loc = idx_to_loc[anc_indices[-num_anc+k]]
        circuit.append('DETECTOR', [stim.target_rec(-num_anc+k)], (*loc, 0))
    circuit.append('SHIFT_COORDS', [], (0,0,1))
    circuit.append('TICK')

    #   logical measurement
    data_qubits = range(d_final ** 2)
    if init_pattern == 'xz':
        if basis == 'X':
            circuit.append('MX', data_qubits)
            circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-k*d_final) for k in range(1, d_final+1)], 0)
        else:
            circuit.append('MZ', data_qubits)
            circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-k) for k in range(d_final * d_final - d_final + 1, d_final * d_final + 1)], 0)
    elif init_pattern == 'general':
        for _idx in data_qubits:
            if _idx == 0:
                circuit.append(f'M{basis}', _idx)
                continue
            j, i = divmod(_idx, d_final)
            if i > j:
                circuit.append('MZ', _idx)
            else: # j >= i
                circuit.append('MX', _idx)
        if basis == 'X':
            logical_lst = [j * d_final for j in range(d_final)]
        elif basis == 'Z':
            logical_lst = list(range(d_final))
        elif basis == 'Y':
            logical_lst = list(range(d_final)) + [j * d_final for j in range(1, d_final)]
        logical_lst = [idx - d_final ** 2 for idx in logical_lst]
        circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(k) for k in logical_lst], 0)
    
    # Finally work a bit on mask
    mask = None
    if do_post_select:
        if init_pattern == 'general':
            num1 = (d0 * d0 - d0) // 2 + (d0 * d0 - 1) * (post_select_rounds - 1)
        elif init_pattern == 'xz':
            num1 = (d0 * d0 - 1) // 2 + (d0 * d0 - 1) * (post_select_rounds - 1)
        num0 = d_final * d_final - 1
        assert num1 + num0 == circuit.num_detectors, f'num0+num1={num1 + num0} but num_detectors={circuit.num_detectors}.'
        mask = np.packbits([1 for _ in range(num1)] + [0 for _ in range(num0)], bitorder='little', axis=0)
    return circuit, mask


if __name__ == '__main__':
    circuit, mask = OscarEncoderCircuit(
        k_final = 3, 
        basis = 'y', 
        p = 1e-3,
        p2 = 1e-3,
        k_init = 1, 
        init_pattern = 'general', 
        post_select_rounds = 3,
        )
    # sampler = circuit.compile_detector_sampler()
    detector_error_model = circuit.detector_error_model(decompose_errors=True)