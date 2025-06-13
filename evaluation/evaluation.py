import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_context("talk")

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('-')

    metadata = {}

    if 'opal-scripts' in filepath:
        metadata['benchmark'] = 'opal'
    else:
        if 'noclean.json' in filename:
            metadata['benchmark'] = parts[-2]
        else:
            metadata['benchmark'] = parts[-1].split('.')[0]

    if 'EX1-P' in filename:
        metadata['analysis'] = 'Purity'
        purity_match = re.search(r'-L([0-2])-(L[0-1]|none)-', filename)
        if purity_match:
            metadata['analysis_level'] = f"L{purity_match.group(1)}"
            metadata['escape_level'] = purity_match.group(2)
        else:
            metadata['analysis_level'], metadata['escape_level'] = 'N/A', 'N/A'
    elif 'EX2-I' in filename:
        metadata['analysis'] = 'Immutability'
        metadata['analysis_level'], metadata['escape_level'] = 'N/A', 'N/A'
    else:
        metadata['analysis'] = 'Unknown'
        metadata['analysis_level'], metadata['escape_level'] = 'N/A', 'N/A'

    metadata['cleanup'] = 'noclean' not in filename
    metadata['compact'] = 'Compact' in filename

    key_parts = [
        metadata['benchmark'],
        metadata['analysis'],
        metadata['analysis_level'],
        metadata['escape_level'],
        str(metadata['cleanup']),
        str(metadata['compact'])
    ]

    counter_match = re.search(r'-(\d+)-Compact|-(\d+)-[a-zA-Z]+|-(\d+)-noclean|(\d+)\.json', filename)
    if counter_match:
        counter = next(g for g in counter_match.groups() if g is not None)
        key_parts.append(counter)

    metadata['key'] = '_'.join(key_parts)
    return metadata

def get_json_metrics(filepath):
    max_heap, gc_count = 0, 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            events = data.get('recording', {}).get('events', [])
            for event in events:
                if event.get('type') == 'jdk.GCHeapSummary':
                    heap_used = event.get('values', {}).get('heapUsed', 0)
                    if heap_used > max_heap: max_heap = heap_used
                elif event.get('type') == 'jdk.GarbageCollection':
                    gc_count += 1
    except (json.JSONDecodeError, FileNotFoundError):
        return None, None
    peak_heap_mb = max_heap / (1024 * 1024) if max_heap > 0 else 0
    return peak_heap_mb, gc_count

def get_runtime_from_log(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        runs = re.split(r'(?m)(?=^#{10,}[\r\n]+^##### RUNNING)', content)
        runtimes = {}
        for run in runs:
            if not run.strip() or 'RUNNING' not in run: continue
            header_match = re.search(r'RUNNING\s+(L[0-2])?\s*(\w+)\s+AGAINST:\s+(\w+)', run)
            is_opal_run = 'AGAINST: opal' in run
            if is_opal_run:
                benchmark, analysis_level = 'opal', 'N/A'
                run_details_match = re.search(r'RUNNING\s+(L[0-2])?\s*(\w+)', run)
                if not run_details_match: continue
                if run_details_match.group(1): analysis_level = run_details_match.group(1)
                analysis = run_details_match.group(2).title()
            elif header_match:
                analysis_level = header_match.group(1) if header_match.group(1) else 'N/A'
                analysis = header_match.group(2).title()
                benchmark = header_match.group(3)
            else: continue

            escape_level = 'N/A'
            if analysis == 'Purity':
                escape_match = re.search(r'\*\s+(\S+)\s+ESCAPE', run)
                if escape_match: escape_level = escape_match.group(1)

            cleanup_match = re.search(r'Cleanup (true|false)', run)
            compact_match = '##### Compact' in run and 'Compact' in run
            counter_match = re.search(r'Iteration (\d+)', run)
            key_parts = [benchmark, analysis, analysis_level, escape_level, str(cleanup_match.group(1) == 'true') if cleanup_match else 'N/A', str(compact_match), counter_match.group(1) if counter_match else 'N/A']
            key = '_'.join(key_parts)

            # MODIFIZIERT: Regex, um beide Formate der "analysis time" zu finden
            time_match = re.search(r'Analysis time:\s*(\d+\.\d+)\s*s|(\d+\.\d+)\s*s seconds analysis time', run)
            if time_match:
                # Nimmt den Wert aus der ersten oder zweiten Gruppe, je nachdem, welche gefunden wurde
                time_str = time_match.group(1) or time_match.group(2)
                if time_str:
                    runtimes[key] = float(time_str)
        return runtimes
    except FileNotFoundError as e:
        print(f"Log file not found: {e}")
        return {}


data_records = []
base_dir = './'
json_dirs = [os.path.join(base_dir, 'jfr-json/dacapo'), os.path.join(base_dir, 'jfr-json/opal-scripts')]
for directory in json_dirs:
    if not os.path.isdir(directory): continue
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            metadata = parse_filename(filepath)
            peak_heap, gc_runs = get_json_metrics(filepath)
            if peak_heap is not None:
                metadata['PeakHeap_MB'], metadata['gc_runs'] = peak_heap, gc_runs
                data_records.append(metadata)

df_heap = pd.DataFrame(data_records)
all_runtimes = {}
log_files = {
    'jfr-json/dacapo/ex1_purity.sanitized.log': 'Purity',
    'jfr-json/dacapo/experiment2_immutability.sanitized.log': 'Immutability',
    'jfr-json/opal-scripts/ex1_purity.sanitized.log': 'Purity',
    'jfr-json/opal-scripts/experiment2_immutability.sanitized.log': 'Immutability'
}
for log_path, _ in log_files.items():
    full_path = os.path.join(base_dir, log_path)
    if os.path.exists(full_path): all_runtimes.update(get_runtime_from_log(full_path))
    else: print(f"Log file does not exist, skipping: {full_path}")

df_heap['Runtime_s'] = df_heap['key'].map(all_runtimes)
df = df_heap.dropna(subset=['PeakHeap_MB', 'Runtime_s']).copy()

if df.empty:
    print("FATAL: No matching data found after merging. Exiting.")
    exit()

df['Cleanup_Label'] = df['cleanup'].apply(lambda x: 'Clean' if x else 'NoClean')
df['Treatment'] = df.apply(lambda r: f"{'Clean' if r['cleanup'] else 'NoClean'}{' + Compact' if r['compact'] else ''}", axis=1)
benchmarks_ordered = sorted(df['benchmark'].unique())
df_ff12 = df[~df['compact']].copy()
df_purity = df_ff12[df_ff12['analysis'] == 'Purity']
df_immutability = df_ff12[df_ff12['analysis'] == 'Immutability']

def generate_ff12_tables(df_in, metric_col, metric_name, unit, include_gc=False, aggregate_levels=False):
    if df_in.empty: return f"Input data for {metric_name} table is empty."

    is_purity = 'escape_level' in df_in.columns and df_in['escape_level'].nunique() > 1

    if is_purity and not aggregate_levels:
        grouping_cols = ['benchmark', 'analysis_level', 'escape_level', 'Cleanup_Label']
    else:
        grouping_cols = ['benchmark', 'Cleanup_Label']

    aggs = {'median_val': (metric_col, 'median'), 'count_val': (metric_col, 'size')}
    if include_gc: aggs['gc_median'] = ('gc_runs', 'median')

    agg_df = df_in.groupby(grouping_cols).agg(**aggs).unstack()

    if is_purity and not aggregate_levels:
        idx_tuples = df_in[['benchmark', 'analysis_level', 'escape_level']].drop_duplicates().sort_values(by=['benchmark', 'analysis_level', 'escape_level'])
        df_out = pd.DataFrame(index=pd.MultiIndex.from_frame(idx_tuples))
    else:
        df_out = pd.DataFrame(index=benchmarks_ordered)

    no_clean_col = f'{metric_name} [{unit}]'
    clean_col_req = 'PeakLaufzeit [s] Cleanup' if metric_name == 'Laufzeit' else f'{metric_name} [MB] Cleanup'

    if ('median_val', 'NoClean') in agg_df.columns: df_out[no_clean_col] = agg_df[('median_val', 'NoClean')]
    if ('count_val', 'NoClean') in agg_df.columns: df_out['N'] = agg_df[('count_val', 'NoClean')]
    if ('median_val', 'Clean') in agg_df.columns: df_out[clean_col_req] = agg_df[('median_val', 'Clean')]
    if ('count_val', 'Clean') in agg_df.columns: df_out['N (Cleanup)'] = agg_df[('count_val', 'Clean')]

    df_out['Improvement [%]'] = ((df_out[no_clean_col] - df_out[clean_col_req]) / df_out[no_clean_col]) * 100

    final_cols = [no_clean_col, 'N', clean_col_req, 'N (Cleanup)']
    if include_gc:
        if ('gc_median', 'NoClean') in agg_df.columns: df_out['# GC-Läufe'] = agg_df[('gc_median', 'NoClean')]
        if ('gc_median', 'Clean') in agg_df.columns: df_out['# GC-Läufe (Cleanup)'] = agg_df[('gc_median', 'Clean')]
        final_cols.extend(['# GC-Läufe', '# GC-Läufe (Cleanup)'])

    final_cols.append('Improvement [%]')
    df_out = df_out.reindex(columns=final_cols).fillna(0)

    formatters = {col: '{:.0f}'.format for col in df_out.columns if col.startswith('N') or col.startswith('# GC')}
    return df_out.to_latex(float_format="%.2f", na_rep='-', formatters=formatters, escape=False)


def generate_ff3_table(df_in, metric_col, metric_name, unit):
    if df_in.empty: return f"Input data for {metric_name} table is empty."
    agg_data = df_in.groupby(['benchmark', 'Treatment']).agg(median_val=(metric_col, 'median'), count_val=(metric_col, 'size')).unstack()
    df_out = pd.DataFrame(index=benchmarks_ordered)
    all_treatments = ['NoClean', 'Clean', 'NoClean + Compact', 'Clean + Compact']
    clean_col_name = f'PeakLaufzeit [s] Cleanup' if metric_name == 'Laufzeit' else f'{metric_name} [{unit}] Cleanup'
    final_col_names = [f'{metric_name} [{unit}]', clean_col_name, f'{metric_name} [{unit}] Compact', f'{metric_name} [{unit}] Compact+Cleanup']
    final_order = []
    for i, treatment in enumerate(all_treatments):
        median_col_key, count_col_key = ('median_val', treatment), ('count_val', treatment)
        final_median_name, final_count_name = final_col_names[i], f'N ({treatment})'
        final_order.extend([final_median_name, final_count_name])
        if median_col_key in agg_data.columns:
            df_out[final_median_name], df_out[final_count_name] = agg_data[median_col_key], agg_data[count_col_key].astype(int)
        else:
            df_out[final_median_name], df_out[final_count_name] = np.nan, 0
    formatters = {col: '{:.0f}'.format for col in df_out.columns if col.startswith('N (')}
    return df_out[final_order].to_latex(float_format="%.2f", na_rep='-', escape=False)


if not df_purity.empty:
    plt.figure(figsize=(16, 8)); sns.boxplot(data=df_purity, x='benchmark', y='PeakHeap_MB', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'lightblue', 'NoClean': 'lightcoral'}, hue_order=['NoClean', 'Clean']); plt.title('FF1: Peak Heap Purity'); plt.xlabel('Benchmark'); plt.ylabel('Peak Heap [MB]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff1_peakheap_purity.png', dpi=300); plt.close()
    fig, ax1 = plt.subplots(figsize=(16, 8)); sns.boxplot(data=df_purity, x='benchmark', y='Runtime_s', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'lightblue', 'NoClean': 'lightcoral'}, hue_order=['NoClean', 'Clean'], ax=ax1); ax1.set_ylabel('Laufzeit [s]'); ax1.set_xlabel('Benchmark'); ax1.tick_params(axis='x', rotation=45); ax2 = ax1.twinx(); sns.boxplot(data=df_purity, x='benchmark', y='gc_runs', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'deepskyblue', 'NoClean': 'red'}, hue_order=['NoClean', 'Clean'], ax=ax2); ax2.set_ylabel('# GC-Läufe'); plt.title('FF2: Laufzeit und GC-Läufe (Purity)'); fig.tight_layout(); handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(handles1, labels1, title='Laufzeit', loc='upper left'); ax2.legend(handles2, labels2, title='GC-Läufe', loc='upper right'); plt.savefig('ff2_runtime_purity.png', dpi=300); plt.close()
if not df_immutability.empty:
    plt.figure(figsize=(12, 7)); sns.boxplot(data=df_immutability, x='benchmark', y='PeakHeap_MB', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'lightblue', 'NoClean': 'lightcoral'}, hue_order=['NoClean', 'Clean']); plt.title('FF1: Peak Heap Immutability'); plt.xlabel('Benchmark'); plt.ylabel('Peak Heap [MB]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff1_peakheap_immutability.png', dpi=300); plt.close()
    fig, ax1 = plt.subplots(figsize=(12, 7)); sns.boxplot(data=df_immutability, x='benchmark', y='Runtime_s', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'lightblue', 'NoClean': 'lightcoral'}, hue_order=['NoClean', 'Clean'], ax=ax1); ax1.set_ylabel('Laufzeit [s]'); ax1.set_xlabel('Benchmark'); ax1.tick_params(axis='x', rotation=45); ax2 = ax1.twinx(); sns.boxplot(data=df_immutability, x='benchmark', y='gc_runs', hue='Cleanup_Label', order=benchmarks_ordered, palette={'Clean': 'deepskyblue', 'NoClean': 'red'}, hue_order=['NoClean', 'Clean'], ax=ax2); ax2.set_ylabel('# GC-Läufe'); plt.title('FF2: Laufzeit und GC-Läufe (Immutability)'); fig.tight_layout(); handles1, labels1 = ax1.get_legend_handles_labels(); handles2, labels2 = ax2.get_legend_handles_labels(); ax1.legend(handles1, labels1, title='Laufzeit', loc='upper left'); ax2.legend(handles2, labels2, title='GC-Läufe', loc='upper right'); plt.savefig('ff2_runtime_immutability.png', dpi=300); plt.close()

df_purity_ff3_all = df[df['analysis'] == 'Purity']
df_immutability_ff3 = df[df['analysis'] == 'Immutability']

if not df_purity_ff3_all.empty:
    plt.figure(figsize=(16, 8)); sns.boxplot(data=df_purity_ff3_all, x='benchmark', y='PeakHeap_MB', hue='Treatment', order=benchmarks_ordered, hue_order=['NoClean', 'Clean', 'NoClean + Compact', 'Clean + Compact'], palette={'NoClean': '#ff9999', 'Clean': '#9999ff', 'NoClean + Compact': '#ff6666', 'Clean + Compact': '#6666ff'}); plt.title('FF3: Peak Heap Purity'); plt.xlabel('Benchmark'); plt.ylabel('Peak Heap [MB]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff3_peakheap_purity.png', dpi=300); plt.close()
    plt.figure(figsize=(16, 8)); sns.boxplot(data=df_purity_ff3_all, x='benchmark', y='Runtime_s', hue='Treatment', order=benchmarks_ordered, hue_order=['NoClean', 'Clean', 'NoClean + Compact', 'Clean + Compact'], palette={'NoClean': '#ff9999', 'Clean': '#9999ff', 'NoClean + Compact': '#ff6666', 'Clean + Compact': '#6666ff'}); plt.title('FF3: Laufzeit Purity'); plt.xlabel('Benchmark'); plt.ylabel('Laufzeit [s]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff3_runtime_purity.png', dpi=300); plt.close()
if not df_immutability_ff3.empty:
    plt.figure(figsize=(16, 8)); sns.boxplot(data=df_immutability_ff3, x='benchmark', y='PeakHeap_MB', hue='Treatment', order=benchmarks_ordered, hue_order=['NoClean', 'Clean', 'NoClean + Compact', 'Clean + Compact'], palette={'NoClean': '#ff9999', 'Clean': '#9999ff', 'NoClean + Compact': '#ff6666', 'Clean + Compact': '#6666ff'}); plt.title('FF3: Peak Heap Immutability'); plt.xlabel('Benchmark'); plt.ylabel('Peak Heap [MB]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff3_peakheap_immutability.png', dpi=300); plt.close()
    plt.figure(figsize=(16, 8)); sns.boxplot(data=df_immutability_ff3, x='benchmark', y='Runtime_s', hue='Treatment', order=benchmarks_ordered, hue_order=['NoClean', 'Clean', 'NoClean + Compact', 'Clean + Compact'], palette={'NoClean': '#ff9999', 'Clean': '#9999ff', 'NoClean + Compact': '#ff6666', 'Clean + Compact': '#6666ff'}); plt.title('FF3: Laufzeit Immutability'); plt.xlabel('Benchmark'); plt.ylabel('Laufzeit [s]'); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig('ff3_runtime_immutability.png', dpi=300); plt.close()

# Tables
table_configs = {
    'ff1_peakheap_purity_table.tex': (generate_ff12_tables, {'df_in': df_purity, 'metric_col': 'PeakHeap_MB', 'metric_name': 'Peak Heap', 'unit': 'MB'}),
    'ff1_peakheap_immutability_table.tex': (generate_ff12_tables, {'df_in': df_immutability, 'metric_col': 'PeakHeap_MB', 'metric_name': 'Peak Heap', 'unit': 'MB'}),
    'ff2_runtime_purity_table.tex': (generate_ff12_tables, {'df_in': df_purity, 'metric_col': 'Runtime_s', 'metric_name': 'Laufzeit', 'unit': 's', 'include_gc': True, 'aggregate_levels': True}),
    'ff2_runtime_purity_table_aufgeschlusselt.tex': (generate_ff12_tables, {'df_in': df_purity, 'metric_col': 'Runtime_s', 'metric_name': 'Laufzeit', 'unit': 's', 'include_gc': True, 'aggregate_levels': False}),
    'ff2_runtime_immutability_table.tex': (generate_ff12_tables, {'df_in': df_immutability, 'metric_col': 'Runtime_s', 'metric_name': 'Laufzeit', 'unit': 's', 'include_gc': True}),
    'ff3_peakheap_purity_table.tex': (generate_ff3_table, {'df_in': df_purity_ff3_all, 'metric_col': 'PeakHeap_MB', 'metric_name': 'Peak Heap', 'unit': 'MB'}),
    'ff3_peakheap_immutability_table.tex': (generate_ff3_table, {'df_in': df_immutability_ff3, 'metric_col': 'PeakHeap_MB', 'metric_name': 'Peak Heap', 'unit': 'MB'}),
    'ff3_runtime_purity_table.tex': (generate_ff3_table, {'df_in': df_purity_ff3_all, 'metric_col': 'Runtime_s', 'metric_name': 'Laufzeit', 'unit': 's'}),
    'ff3_runtime_immutability_table.tex': (generate_ff3_table, {'df_in': df_immutability_ff3, 'metric_col': 'Runtime_s', 'metric_name': 'Laufzeit', 'unit': 's'})
}

for filename, (func, kwargs) in table_configs.items():
    print(f"Generating table: {filename}...")
    latex_string = func(**kwargs)
    with open(filename, 'w', encoding='utf-8') as f: f.write(latex_string)
