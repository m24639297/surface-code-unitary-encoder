import  sinter
from stim_my_encoder import MyEncoderCircuit
from stim_oscar_encoder import OscarEncoderCircuit
from stim_normal_encoder import NormalEncoderCircuit

outfile_path = './stats.csv'

ki = 1
di = 2 ** ki + 1
kfs = [1,2,3,4]
basis = 'Y'
_ps = [1e-3, 5e-3, 1e-2]
ps  = [1 * p for p in _ps]
p2s = [5 * p for p in _ps] 
pms = [5 * p for p in _ps]
post_selection_rounds = [2]
 
# For normal encoder
msmt_rounds = [2]
dfs = [3,5,9,17]

tasks = []
## my encoder
tasks += \
[
    sinter.Task(
        circuit = MyEncoderCircuit(
            k_final = kf, 
            basis = basis, 
            p = p,
            p2 = p2,
            pm = pm,
            k_init = ki, 
            init_pattern = 'general', 
            post_select_rounds = psr,
        )[0],
        postselection_mask = MyEncoderCircuit(
            k_final = kf, 
            basis = basis, 
            p = p,
            p2 = p2,
            pm = pm,
            k_init = ki, 
            init_pattern = 'general', 
            post_select_rounds = psr,
        )[1],
        json_metadata = {'encoder': 'mine',
                    'd_init': di,
                    'd_final': 2 ** kf + 1,
                    'p': p,
                    'p2': p2,
                    'pm': pm,
                    'psr': psr}
    )
  for kf in kfs
  for p, p2, pm in zip(ps, p2s, pms)
  for psr in post_selection_rounds
]

## + local encoder
tasks += \
[
    sinter.Task(
        circuit = OscarEncoderCircuit(
            k_final = 2 ** (kf - 1), 
            basis = basis, 
            p = p,
            p2 = p2,
            pm = pm,
            k_init = ki, 
            init_pattern = 'general', 
            post_select_rounds = psr,
        )[0],
        postselection_mask = OscarEncoderCircuit(
            k_final = 2 ** (kf - 1), 
            basis = basis, 
            p = p,
            p2 = p2,
            pm = pm,
            k_init = ki, 
            init_pattern = 'general', 
            post_select_rounds = psr,
        )[1],
        json_metadata = {'encoder': 'local',
                    'd_init': di,
                    'd_final': 2 ** kf + 1,
                    'p': p,
                    'p2': p2,
                    'pm': pm,
                    'psr': psr}
    )
  for kf in kfs
  for p, p2, pm in zip(ps, p2s, pms)
  for psr in post_selection_rounds
]

## + normal encoder
tasks += \
[
    sinter.Task(
        circuit = NormalEncoderCircuit(
          d_final = df, 
          basis = basis, 
          p = p,
          msmt_rounds=msmtr,
          p2 = p2,
          pm = pm,
          post_select_rounds = psr,
        )[0],
        postselection_mask = NormalEncoderCircuit(
          d_final = df, 
          basis = basis, 
          p = p,
          msmt_rounds=msmtr,
          p2 = p2,
          pm = pm,
          post_select_rounds = psr,
        )[1],
        json_metadata = {'encoder': f'normal{msmtr}',
                    'd_init': 3,
                    'd_final': df,
                    'p': p,
                    'p2': p2,
                    'pm': pm,
                    'psr': psr,}
    )
  for df in dfs
  for p, p2, pm in zip(ps, p2s, pms)
  for psr in post_selection_rounds
  for msmtr in msmt_rounds
]

samples = sinter.collect(
    num_workers=4,
    tasks=tasks,
    decoders=['pymatching'],
    max_shots=2000000,
    max_errors=10000,
    save_resume_filepath='outfile_path',
)