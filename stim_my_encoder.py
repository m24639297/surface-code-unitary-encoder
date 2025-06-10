import itertools, stim
from typing import Optional, Tuple, List
import numpy as np
import tools as sc

r""" Surface code layout & gate scheduling(1~4)

(XL)               X
 j
 ↑     + ───── + ──────+    -
 |     │1     2│1      │     |
 |  Z  │   X   │   Z   │     | a
 |     │       │2      │     |
 |     + ───── + ───── 0    -
 |     │1      │1     2│1
 |     │   Z   │   X   │  Z
 |     │2      │       │2
 |     o ───── 0 ───── 0
 |      1     2
           X
     --------------------> i (ZL)
          
Initial pattern for rot code: 
    bot-left qubit (o): |psi>
    NE-SW diagonal and to its right (0): |0>
    the rest (x): |+>
    Fixed X stab: j > i
    Fixed Z stab: j < i
                

      ·───X───·───X───·    -
      │ \   /   \   / │     |
      Z   ·   Z   ·   Z     | a
      │ /   \   /   \ │     |
      ·   X   ·   X   ·    -
      │ \   /   \   / │
      Z   ·   Z   ·   Z 
      │ /   \   /   \ │
      ·───X───·───X───·

    w/   
           3             3     
         /   \         /   \   
       4   Z   ·     ·   X   4 
         \   /         \   /   
           ·             ·                
"""

def rot_doubling_gate_seq(d: int, 
                          a: int, 
                          loc_to_idx: Optional[dict]=None, 
                          flattened: bool=True,):
    return sc.get_rot_gate_seq(d, a, loc_to_idx, flattened, bulk_only=True)[0:2] + sc.get_reg_gate_seq(d, a, loc_to_idx, flattened)[0:2]
 
def MyEncoderCircuit(
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
        rotated surface code with my new encoder; `mask` is a list of np.uint8
        (bit-packed boolean array for whether post-select each detector, in
        little endian) for the `postselection_mask` arg in sinter.Task
        initializer. `mask` is None if `post_select_rounds` is 0, i.e. without
        post-selection. 
    """
    MODES = {
        'data_in_use',
        'circuit',
    }
    PATTERNS = {
        'xz',
        'general',
        'general_li'
    }
    basis = basis.upper()
    assert k_final >= k_init
    assert init_pattern in PATTERNS, f'init_pattern must be one of {PATTERNS}.'
    assert basis in 'XYZ', f'basis must be one of X, Y, Z.'
    if idle_per_gate and idle_per_round:
        print('[Warning] Both `idle_per_gate` and `idle_per_round` are `True`.')
    if basis == 'Y':
        assert init_pattern == 'general'

    p_init = p if p_init is None else p_init
    p2 = p if p2 is None else p2
    pm = p if pm is None else pm

    loc_to_idx = dict()
    idx_to_loc = list()
    d_final = 2 ** k_final + 1
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
    # k=k_init
    if k_final > k_init:
        for loc in sc.get_rot_boundary_anc_locs(2**k_init + 1, 2**(k_final - k_init)):
            idx_to_loc.append(loc)
            loc_to_idx[loc] = idx
            idx += 1
    del idx

    convert_locs_to_indices = lambda locs: [loc_to_idx[loc] for loc in locs]
    d0 = 2 ** k_init + 1
    a0 = 2 ** (k_final - k_init)
    data0 = convert_locs_to_indices(sc.get_rot_data_locs(d0, a0))
    anc0 = convert_locs_to_indices(sc.get_rot_anc_locs(d0, a0))
    x_anc0 = convert_locs_to_indices(sc.get_rot_x_anc_locs(d0, a0))  
    all_qb0 = convert_locs_to_indices(sc.get_rot_all_locs(d0, a0))

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
        logical_idx = loc_to_idx[(0, 0)]
        init_dict['X'] = convert_locs_to_indices(sc.get_rot_init_in_x_locs(d0, a0))
        init_dict['Z'] = convert_locs_to_indices(sc.get_rot_init_in_z_locs(d0, a0))
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
        gate_seq = sc.get_rot_gate_seq(d0, a0, loc_to_idx)
        for sub_idx, gates in enumerate(gate_seq):
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

        for loc in sc.get_rot_x_anc_locs(d0, a0):
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
                gauge_locs = sc.get_z_gauge_locs(d0, a0, loc)
                gate_args = list(itertools.chain(*[(stim.target_rec(-1), _idx) for _idx in convert_locs_to_indices(gauge_locs)]))
                circuit.append('CZ', gate_args)

        for loc in sc.get_rot_z_anc_locs(d0, a0):
            anc_idx = loc_to_idx[loc]
            circuit.append('DEPOLARIZE1', anc_idx, pm0)
            circuit.append('MR', anc_idx)
            is_fixed = (init_pattern == 'xz' and basis == 'Z') or (init_pattern == 'general' and loc[0] > loc[1])
            if _idx == 0 and is_fixed:
                circuit.append('DETECTOR', [stim.target_rec(-1)], (*loc, 0))
            if _idx > 0:
                circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-d0*d0)], (*loc, 0))
            if _idx == post_select_rounds - 1:
                gauge_locs = sc.get_x_gauge_locs(d0, a0, loc)
                gate_args = list(itertools.chain(*[(stim.target_rec(-1), _idx) for _idx in convert_locs_to_indices(gauge_locs)]))
                circuit.append('CX', gate_args)

        circuit.append('SHIFT_COORDS', [], (0,0,1))
        circuit.append('TICK')

    # Expanding
    for k in range(k_init, k_final):
        a = 2 ** (k_final - k)
        d = 2 ** k + 1
        # This includes all data qubits at this point + all fresh qubits for building the next round
        all_qubits_this_round = convert_locs_to_indices(sc.get_rot_data_locs(2*d-1, a//2))
        if idle_per_round:
            circuit.append('DEPOLARIZE1', all_qubits_this_round, p)
            circuit.append('TICK')
        if init_error:
            fresh_qubits = sc.get_rot_bulk_anc_locs(d, a) + sc.get_reg_anc_locs(d, a)
            fresh_qubits = convert_locs_to_indices(fresh_qubits)
            circuit.append('DEPOLARIZE1', fresh_qubits, p_init)
        circuit.append('H', convert_locs_to_indices(sc.get_rot_bulk_x_anc_locs(d, a)))
        circuit.append('H', convert_locs_to_indices(sc.get_reg_x_anc_locs(d, a)))
        for gates in rot_doubling_gate_seq(d, a, loc_to_idx):
            circuit.append('CX', gates)
            if gate_error:
                circuit.append('DEPOLARIZE2', gates, p2)
            if idle_per_gate:
                idle_qubits = [q for q in all_qubits_this_round if q not in set(gates)]
                circuit.append('DEPOLARIZE1', idle_qubits, p)
            circuit.append('TICK')
            
    # data qubit error on final code
    if last_data_error: 
        circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_data_locs(d_final, 1)), p)

    # one round of perfect stabilizer measurements
    x_anc_indices = convert_locs_to_indices(sc.get_rot_x_anc_locs(d_final, 1))
    anc_indices = convert_locs_to_indices(sc.get_rot_anc_locs(d_final, 1))
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

    circuit, mask = MyEncoderCircuit(
        k_final = 2, 
        basis = 'x', 
        p = 1e-3,
        p2 = 1e-3,
        k_init = 1, 
        init_pattern = 'general', 
        post_select_rounds = 0,
        idle_per_gate = True,
        idle_per_round = False,
        gate_error = True,
        init_error = True
        )

    # sampler = circuit.compile_detector_sampler()
    detector_error_model = circuit.detector_error_model(decompose_errors=True)