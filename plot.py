import sinter
import numpy as np
import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Set which fig to plot here (0: p1/p2 ratio, 1: decoder comparison)
idx = 0

outfile_path = [
    'p1,p2_ratio_comparison.pdf',
    'encoders_bar_plot_comparison.pdf',
][idx]
stat_file = [
    './paper_data/noise_ratio_sim.csv', # for p1/p2 ratio plot, Fig.6
    './paper_data/encoder_comparison.csv', # for encoder comparison plot, Fig.7
][idx]
samples = sinter.read_stats_from_csv_files(stat_file)

if idx == 0:
    tmp = {
    'local': 'local',
    'mine': 'nonlocal'
    }
    def plot_args_func(
        curve_index: int,  # a unique incrementing integer for each curve
        curve_group_key: str,  # what group_func returned
        stats: list,  # the data points on the curve
    ) -> dict:
        stat = stats[0]
        params = dict()
        if stat.json_metadata['d_final'] == 5:
            params['color'] = colors[0]
        if stat.json_metadata['d_final'] == 9:
            params['color'] = colors[1]
        if stat.json_metadata['d_final'] == 17:
            params['color'] = colors[2]

        if stat.json_metadata['encoder'] == 'local':
            params['linestyle'] = ':'
        if stat.json_metadata['encoder'] == 'mine':
            params['linestyle'] = '-'
        params['marker'] = 'o'
        return params

    plt.rcParams.update({'font.size': 12})
    plt.rcParams['lines.markersize'] = 5
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f'{tmp[stat.json_metadata["encoder"]]}, {stat.json_metadata["d_final"]}',
        x_func=lambda stat: stat.json_metadata['p']/stat.json_metadata['p2'],         
        plot_args_func = plot_args_func             
    )
    ax.semilogy()
    ax.grid()
    ax.set_ylabel('Logical Error Rate')
    ax.set_xlabel(r'$p_1/p_2$')
    ax.legend(loc='upper left',)
    plt.tight_layout()
    plt.savefig(outfile_path, format='pdf', bbox_inches="tight")

if idx == 1:
    def get_time_ovhd(stat,):
        cz_time = 1
        msmt_time = 10
        encoder = stat.json_metadata['encoder']
        p_succ = 1 - stat.discards / stat.shots
        d_final = stat.json_metadata['d_final']
        d_init = stat.json_metadata['d_init']
        time = 2 * (4 * cz_time + msmt_time) / p_succ
        rounds = 0
        if encoder == 'local':
            rounds = (d_final - d_init) // 2
        if encoder == 'mine':
            rounds = int(np.log2(d_final - 1)) - 1
        if encoder.startswith('normal'):
            if d_final > 3:
                rounds = int(encoder[6:])
                time += rounds * msmt_time
        time += (rounds * 4 * cz_time)
        return time
    
    ds = [3, 5, 9, 17]
    results = {
        'mine':[[None for _ in range(len(ds))] for _ in range(3)],
        'local': [[None for _ in range(len(ds))] for _ in range(3)],
        'normal2':[[None for _ in range(len(ds))] for _ in range(3)],
    }
    for stat in samples:
        meta = stat.json_metadata
        if (meta['p'] == 0.001 
        and meta['p2'] == 0.005 
        and meta['pm'] == 0.005):
            key = meta['encoder']
            idx = ds.index(meta['d_final'])
            x = get_time_ovhd(stat)
            y = stat.errors / (stat.shots - stat.discards)
            yerr = np.sqrt(y * (1-y) / (stat.shots - stat.discards))
            results[key][0][idx] = x
            results[key][1][idx] = y
            results[key][2][idx] = yerr

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    width = 0.3
    ds = np.array([3,5,9,17])
    indices = np.arange(len(ds))
    axes[0].bar(indices - width, results['mine'][1] , width, yerr = results['mine'][2], color = colors[0], label='non-local', log=True)
    axes[0].bar(indices, results['local'][1], width, yerr = results['local'][2], color = colors[1], label='local', log=True)
    axes[0].bar(indices + width, results['normal2'][1], width, yerr = results['normal2'][2], color = colors[2], label='msmt', log=True)

    axes[0].set_xticks(indices)
    axes[0].set_xticklabels([f'{d}' for d in ds])
    axes[0].set_xlabel(r'final code distance $d_\mathrm{f}$')
    axes[0].set_title('Logical error rate',fontsize=10)
    axes[0].set_ylim((5e-3, 3e-2))
    axes[0].legend()

    axes[1].bar(indices - width, results['mine'][0], width, color = colors[0], label='non-local')
    axes[1].bar(indices, results['local'][0], width, color = colors[1], label='local')
    axes[1].bar(indices + width, results['normal2'][0], width, color = colors[2], label='msmt')

    axes[1].set_xticks(indices)
    axes[1].set_xticklabels([f'{d}' for d in ds])
    axes[1].set_xlabel(r'final code distance $d_\mathrm{f}$')
    axes[1].set_title('Expected time per kept shot',fontsize=10)
    axes[1].set_ylim((20, 72))
    axes[1].axhline(y=results['mine'][0][0], color='red', linestyle='--', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(outfile_path, format='pdf', bbox_inches="tight")