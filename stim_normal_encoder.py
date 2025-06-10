import itertools, stim
from typing import Optional, Tuple, List
import numpy as np
import tools as sc

r""" Surface code layout & gate scheduling(1~4)

      grow 
(XL)   |           X
 j     |
 ↑     + ───── + ──────+    -
 |     │1     2│1      │     |
 |  Z  │   X   │   Z   │     | a=1
 |     │       │2      │     |
 |     + ───── + ───── 0    -
 |     │1      │1     2│1
 |     │   Z   │   X   │  Z
 |     │2      │       │2
 |     o ───── 0 ───── 0 ------ grow
 |      1     2
           X
     --------------------> i (ZL)           
"""

def rot_doubling_gate_seq(d: int, 
                          a: int, 
                          loc_to_idx: Optional[dict]=None, 
                          flattened: bool=True,):
    return sc.get_rot_gate_seq(d, a, loc_to_idx, flattened, bulk_only=True)[0:2] + sc.get_reg_gate_seq(d, a, loc_to_idx, flattened)[0:2]
 
def NormalEncoderCircuit(
        d_final: int, 
        basis: str, 
        p: float, 
        msmt_rounds: int,
        d_init: int = 3, 
        post_select_rounds : Optional[int] = None,
        p_init: float = None,
        p2: float = None,
        pm: float = None,
        idle_error: bool = True,
        init_error: bool = True,
        gate_error: bool = True,
        last_data_error: bool = True,
        msmt_error: bool = True,
        first_round_optimization: bool = True,
        test_xz: bool = False,
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

    basis = basis.upper()
    assert d_final >= d_init
    assert basis in 'XYZ', f'basis must be one of X, Y, Z.'

    if d_final == d_init:
        msmt_rounds = 0
        print(' d_final == d_init, no extra msmt_rounds.')

    p_init = p if p_init is None else p_init
    p2 = p if p2 is None else p2
    pm = p if pm is None else pm

    loc_to_idx = dict()
    idx_to_loc = list()
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

    convert_locs_to_indices = lambda locs: [loc_to_idx[loc] for loc in locs]
    d0 = d_init
    a0 = 1
    data0 = convert_locs_to_indices(sc.get_rot_data_locs(d0, a0))
    anc0 = convert_locs_to_indices(sc.get_rot_anc_locs(d0, a0))
    x_anc0 = convert_locs_to_indices(sc.get_rot_x_anc_locs(d0, a0))  
    all_qb0 = convert_locs_to_indices(sc.get_rot_all_locs(d0, a0))

    circuit = stim.Circuit()
    for idx, loc in enumerate(idx_to_loc):
        circuit.append("QUBIT_COORDS", idx, loc)

    # data initial pattern
    init_dict = {'X': [], 'Z': [], 'Y': []}
    logical_idx = loc_to_idx[(0, 0)]
    init_dict['X'] = convert_locs_to_indices(sc.get_rot_init_in_x_locs(d_final, a0))
    init_dict['Z'] = convert_locs_to_indices(sc.get_rot_init_in_z_locs(d_final, a0))
    init_dict[basis] += [logical_idx]
    if test_xz:
        assert not basis == 'Y', 'cant use Y here'
        init_dict = {'X': [], 'Z': [], 'Y': []}
        init_dict[basis] = convert_locs_to_indices(sc.get_rot_data_locs(d_final, a0))
    circuit.append('H', init_dict['X'])
    circuit.append('H', init_dict['Y'])
    circuit.append('S', init_dict['Y'])
        
    # First few rounds of msmt for post selection and gauge fixing
    p0, p_init0, p20, pm0 = (p, p_init, p2, pm) if do_post_select else (0, 0, 0, 0)   
    untouched_data0 = set(data0)  
    if init_error:
        circuit.append('DEPOLARIZE1', data0, p_init0)
    for _idx in range(post_select_rounds):
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
            if idle_error:
                idle_qubits = [q for q in all_qb0 if q not in set(gates)]
                circuit.append('DEPOLARIZE1', idle_qubits, p0)
            circuit.append('TICK')

        for loc in sc.get_rot_x_anc_locs(d0, a0):
            anc_idx = loc_to_idx[loc]
            circuit.append('H', anc_idx)
            circuit.append('DEPOLARIZE1', anc_idx, pm0)
            circuit.append('MR', anc_idx)
            is_fixed = loc[1] > loc[0]
            if _idx == 0 and is_fixed:
                circuit.append('DETECTOR', [stim.target_rec(-1)], (*loc, 0))
            if _idx > 0:
                circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-d0*d0)], (*loc, 0))
        for loc in sc.get_rot_z_anc_locs(d0, a0):
            anc_idx = loc_to_idx[loc]
            circuit.append('DEPOLARIZE1', anc_idx, pm0)
            circuit.append('MR', anc_idx)
            is_fixed = loc[1] < loc[0]
            if _idx == 0 and is_fixed:
                circuit.append('DETECTOR', [stim.target_rec(-1)], (*loc, 0))
            if _idx > 0:
                circuit.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-d0*d0)], (*loc, 0))
        circuit.append('SHIFT_COORDS', [], (0,0,1))
        circuit.append('TICK')

    # Measure the final code
    all_qb = convert_locs_to_indices(sc.get_rot_all_locs(d_final, 1))
    small_anc_locs = sc.get_rot_anc_locs(d_init, 1)
    new_anc_locs = [loc for loc in sc.get_rot_anc_locs(d_final, 1) if loc not in small_anc_locs]
    new_fixed_locs = [loc for loc in sc.get_rot_x_anc_locs(d_final, 1) if loc[1] > loc[0]] + \
                     [loc for loc in sc.get_rot_z_anc_locs(d_final, 1) if loc[1] < loc[0]]
    n_small = len(small_anc_locs)
    n_new = len(new_anc_locs)
    n_stab = n_small + n_new
    ordered_anc = convert_locs_to_indices(small_anc_locs + new_anc_locs)
    is_fixed_anc = [False for _ in small_anc_locs] + [loc in new_fixed_locs for loc in new_anc_locs]

    x_anc = convert_locs_to_indices(sc.get_rot_x_anc_locs(d_final, 1))
    for m_idx in range(msmt_rounds):
        circuit.append('H', x_anc)
        gate_seq = sc.get_rot_gate_seq(d_final, 1, loc_to_idx)
        for sub_idx, gates in enumerate(gate_seq):
            circuit.append('CX', gates)
            if gate_error:
                circuit.append('DEPOLARIZE2', gates, p2)
            if idle_error:
                idle_qubits = [q for q in all_qb if q not in set(gates)]
                circuit.append('DEPOLARIZE1', idle_qubits, p)
            circuit.append('TICK')
        circuit.append('H', x_anc)
        if idle_error:
            circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_data_locs(d_final, 1)), p)
        if msmt_error:
            circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_anc_locs(d_final, 1)), pm)
        circuit.append('MR', ordered_anc)
        if m_idx == 0:
            for j in range(1, n_new + 1):
                if is_fixed_anc[-j]:
                    circuit.append('DETECTOR', stim.target_rec(-j), (*idx_to_loc[ordered_anc[-j]], 0))
            for j in range(n_new + 1, n_stab + 1):
                circuit.append('DETECTOR', [stim.target_rec(-j), stim.target_rec(-j - n_small)], (*idx_to_loc[ordered_anc[-j]], 0))
        else:
            for j in range(1, n_stab + 1):
                circuit.append('DETECTOR', [stim.target_rec(-j), stim.target_rec(-j - n_stab)], (*idx_to_loc[ordered_anc[-j]], 0))
        circuit.append('SHIFT_COORDS', [], (0,0,1))
        circuit.append('TICK')

    # data qubit error on final code
    if last_data_error: 
        circuit.append('DEPOLARIZE1', convert_locs_to_indices(sc.get_rot_data_locs(d_final, 1)), p)

    # one round of perfect stabilizer measurements
    circuit.append('H', x_anc)
    for gates in sc.get_rot_gate_seq(d_final, 1, loc_to_idx):
        circuit.append('CX', gates)
        circuit.append('TICK')
    circuit.append('H', x_anc)
    circuit.append('MR', ordered_anc)
    for j in range(1, n_stab + 1):
        circuit.append('DETECTOR', [stim.target_rec(-j), stim.target_rec(-j - n_stab)], (*idx_to_loc[ordered_anc[-j]], 0))
    circuit.append('SHIFT_COORDS', [], (0,0,1))
    circuit.append('TICK')

    #   logical measurement
    data_qubits = range(d_final ** 2)
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
    if not test_xz:
        circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(k) for k in logical_lst], 0)
    else:
        if basis == 'Z':
            circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-k) for k in range(1, d_final+1)], 0)
        if basis == 'X':
            circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-k * d_final) for k in range(1, d_final+1)], 0)
    
    # Finally work a bit on mask
    mask = None
    if do_post_select:
        num1 = (d0 * d0 - d0) // 2 + (d0 * d0 - 1) * (post_select_rounds - 1)
        num0 = circuit.num_detectors - num1
        mask = np.packbits([1 for _ in range(num1)] + [0 for _ in range(num0)], bitorder='little', axis=0)
    return circuit, mask


if __name__ == '__main__':

    circuit, mask = NormalEncoderCircuit(
        d_final = 3, 
        basis = 'y', 
        p = 1e-3,
        msmt_rounds=1,
        p2 = 1e-3,
        d_init = 3, 
        post_select_rounds = None,
        gate_error = True,
        init_error = True
        )

    # sampler = circuit.compile_detector_sampler()
    detector_error_model = circuit.detector_error_model(decompose_errors=True)