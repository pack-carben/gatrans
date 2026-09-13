"""Plot original publisher values beside GATrans results, with explicit scope labels."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_sources import read_tables

OMICS = ['SNV', 'METH', 'GE', 'CNA']
COLORS = ['#517bb2', '#8cbbd4', '#f3d88c', '#d95847']
RUN_LABEL = ''


def tidy(table):
    d = table.iloc[1:, 1:].copy()
    d.index = table.iloc[1:, 0].tolist()
    d.columns = table.iloc[0, 1:].tolist()
    return d.astype(float)


def save(fig, out, name):
    if RUN_LABEL:
        fig.text(.5, .005, RUN_LABEL, ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(out / (name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(out / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def heat(ax, frame, title, signed=False, limit=None):
    values = frame.to_numpy(dtype=float)
    if signed:
        limit = limit or max(float(np.nanmax(np.abs(values))), 1e-8)
        im = ax.imshow(np.ma.masked_invalid(values), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    else:
        im = ax.imshow(np.ma.masked_invalid(values), cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(frame.columns)), frame.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(frame)), frame.index)
    ax.set_title(title, fontsize=10)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='results/cancer_specific_v1')
    p.add_argument('--sources', default='results/paper_sources')
    args = p.parse_args()
    run, sources = Path(args.run), Path(args.sources)
    global RUN_LABEL
    configurations = [json.loads(p.read_text()) for p in run.glob('*/*/run_config.json')]
    pilot = any(c['folds'] < 10 or c['epochs'] < 100 for c in configurations)
    RUN_LABEL = ('PILOT - pipeline validation only. ' if pilot else 'Fixed-configuration GATrans evaluation. ') + run.name
    out = run / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    paper2 = read_tables(sources / '41551_2024_1312_MOESM4_ESM.xlsx')
    paper3 = read_tables(sources / '41551_2024_1312_MOESM5_ESM.xlsx')
    metrics, prediction_frames, shap_frames, attention_frames = [], [], [], []
    for kind in ('homogeneous', 'heterogeneous'):
        for dest in sorted((run / kind).glob('*')):
            if not dest.is_dir():
                continue
            for path in dest.glob('fold_*/metrics.json'):
                metrics.append(dict(kind=kind, cancer=dest.name, **json.loads(path.read_text())))
            for filename, collection in [('predictions.csv', prediction_frames), ('shap_genes.csv', shap_frames), ('attention.csv', attention_frames)]:
                if (dest / filename).exists():
                    frame = pd.read_csv(dest / filename)
                    frame['kind'], frame['cancer'] = kind, dest.name
                    collection.append(frame)
    if not metrics:
        raise ValueError('No evaluated models to plot')
    metric = pd.DataFrame(metrics)
    metric.to_csv(out / 'fold_metrics.csv', index=False)
    comparisons = []
    for kind in ('homogeneous', 'heterogeneous'):
        reference = tidy(paper2['FIG2b_' + kind])
        reference.to_csv(out / ('paper_fig2b_' + kind + '.csv'), index_label='cancer')
        ours = metric[metric.kind == kind].groupby('cancer').average_precision.agg(['mean', 'std', 'count'])
        if ours.empty:
            continue
        matched = reference.join(ours, how='inner')
        for cancer, row in matched.iterrows():
            comparisons.append(dict(kind=kind, cancer=cancer, paper_tree=row.TREE, gatrans_ap=row['mean'],
                                    fold_sd=row['std'], folds=int(row['count']), delta_ap=row['mean']-row.TREE))
        fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
        methods = list(reference.columns)
        axs[0].boxplot([reference[m] for m in methods] + [ours['mean']],
                       tick_labels=methods + ['GATrans'], showmeans=True, whis=(0, 100))
        axs[0].set_ylabel('AUPRC (paper) / average precision (GATrans)')
        axs[0].set_title(f'Fig. 2b: {kind}\nPaper: {len(reference)} networks; GATrans: {len(ours)} networks')
        x = np.arange(len(matched))
        axs[1].bar(x-.18, matched.TREE, width=.36, label='Paper TREE', color='#a5afbd')
        axs[1].bar(x+.18, matched['mean'], width=.36, yerr=matched['std'].fillna(0), label='GATrans (fold mean ± SD)', color='#258b8a')
        axs[1].set_xticks(x, matched.index, rotation=60, ha='right')
        axs[1].set_ylim(0, 1)
        axs[1].legend(fontsize=8)
        axs[1].set_title('Matched cancer networks; identical names, new training runs')
        save(fig, out, 'fig2b_' + kind)
    pd.DataFrame(comparisons).to_csv(out / 'paper_vs_gatrans.csv', index=False)
    if not prediction_frames:
        return
    predictions = pd.concat(prediction_frames, ignore_index=True)
    tests = predictions[predictions.split == 'test']
    for kind, group in tests.groupby('kind'):
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        for cancer, data in group.groupby('cancer'):
            fpr, tpr, _ = roc_curve(data.label, data.prediction)
            pr, re, _ = precision_recall_curve(data.label, data.prediction)
            axs[0].plot(fpr, tpr, label=cancer, lw=1)
            axs[1].plot(re, pr, label=cancer, lw=1)
        axs[0].plot([0,1], [0,1], '--', color='gray', lw=.7)
        axs[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Held-out ROC (fold ensemble)')
        axs[1].set(xlabel='Recall', ylabel='Precision', title='Held-out precision-recall')
        axs[1].legend(fontsize=6, ncol=2)
        values = [group.loc[group.label == i, 'prediction'] for i in (0, 1)]
        axs[2].violinplot(values, showmedians=True)
        axs[2].set_xticks([1,2], ['Non-cancer', 'Cancer'])
        axs[2].set(ylabel='Prediction probability', title='Held-out gene-network observations', ylim=(0,1))
        fig.suptitle(kind + ': additional validation plots')
        save(fig, out, 'validation_' + kind)
        # Descriptive cancer specificity: probability spread across cancer networks.
        unlabelled = predictions[(predictions.kind == kind) & (predictions.split == 'unlabelled')]
        matrix = unlabelled.pivot(index='gene', columns='cancer', values='prediction')
        matrix = matrix[matrix.notna().sum(axis=1) >= 2]
        if len(matrix):
            spread = matrix.max(axis=1) - matrix.min(axis=1)
            selected = spread.sort_values(ascending=False).head(20).index
            selected_matrix = matrix.loc[selected]
            selected_matrix.to_csv(out / ('specific_candidates_' + kind + '.csv'))
            fig, ax = plt.subplots(figsize=(max(6, .45*len(matrix.columns)), 7))
            im = heat(ax, selected_matrix, 'Exploratory candidates ranked by cross-cancer score range\nUnlabelled only; missing networks are blank')
            fig.colorbar(im, ax=ax, label='Mean model probability')
            save(fig, out, 'specific_candidates_' + kind)
    if shap_frames:
        shap_data = pd.concat(shap_frames, ignore_index=True)
        shap_data.to_csv(out / 'all_shap_gene_records.csv', index=False)
        ref = tidy(paper3['FIG3a_cancer_specific'])[OMICS]
        ref.to_csv(out / 'paper_fig3a_cancer_specific.csv', index_label='cancer')
        correct = shap_data[shap_data.cohort == 'test_true_positive']
        for kind, group in correct.groupby('kind'):
            v = group.groupby('cancer')[[f'shap_{o}' for o in OMICS]].agg(lambda x: np.abs(x).mean())
            v.columns = OMICS
            v = v.div(v.sum(axis=1).replace(0, np.nan), axis=0)
            v.to_csv(out / ('omics_importance_' + kind + '.csv'))
            order = ref.index.tolist()
            fig, axs = plt.subplots(1, 3, figsize=(13, 6), gridspec_kw={'width_ratios':[1,1,1.3]})
            for ax, values, title in [(axs[0], ref, 'Paper Fig. 3a cancer-specific'), (axs[1], v.reindex(order), f'GATrans {kind} (fold 0)')]:
                left = np.zeros(len(values))
                for omic, color in zip(OMICS, COLORS):
                    ax.barh(np.arange(len(order)), values[omic], left=left, color=color, label=omic)
                    left += values[omic].fillna(0).to_numpy()
                ax.set_yticks(np.arange(len(order)), order)
                ax.set_ylim(-.5, len(order)-.5)
                ax.invert_yaxis()
                ax.set(xlim=(0,1), xlabel='Normalized omics importance', title=title)
            delta = v.reindex(order) - ref
            im = heat(axs[2], delta, 'Difference (descriptive)\nSHAP cohort/aggregation may differ', True, 1)
            fig.colorbar(im, ax=axs[2], fraction=.05)
            axs[0].legend(fontsize=7, loc='lower right')
            save(fig, out, 'fig3a_' + kind)
            delta.to_csv(out / ('fig3a_difference_' + kind + '.csv'))
            dominant = group[[f'shap_{o}' for o in OMICS]].abs().to_numpy().argmax(axis=1)
            counts = np.bincount(dominant, minlength=4)
            original = tidy(paper3['FIG3b_cancer_specific']).iloc[:, 0].reindex(OMICS)
            pd.DataFrame({'omics':OMICS, 'paper_count':original.to_numpy(), 'gatrans_count':counts}).to_csv(out / ('fig3b_counts_' + kind + '.csv'), index=False)
            fig, axs = plt.subplots(1, 3, figsize=(12,4))
            axs[0].pie(original, labels=OMICS, colors=COLORS, wedgeprops={'width':.45}, autopct='%1.0f%%')
            axs[0].set_title(f'Paper Fig. 3b (n={int(original.sum())})')
            axs[1].pie(counts, labels=OMICS, colors=COLORS, wedgeprops={'width':.45}, autopct='%1.0f%%')
            axs[1].set_title(f'GATrans test true positives (n={counts.sum()})')
            for j in range(4):
                scores = group.loc[dominant == j, 'prediction'].to_numpy()
                if len(scores) > 1 and np.ptp(scores) > 0:
                    axs[2].violinplot(scores, positions=[j], showmedians=True)
                elif len(scores):
                    axs[2].scatter([j]*len(scores), scores, color=COLORS[j])
            axs[2].set_xticks(range(4), OMICS)
            axs[2].set(ylim=(0,1), ylabel='Prediction probability', title='New confidence distributions')
            fig.suptitle(kind + ': cohort sizes differ; counts are not a replication tolerance')
            save(fig, out, 'fig3b_' + kind)
        for gene in ('MUC1','KLF6','BAP1','CASP8'):
            paper = tidy(paper3['FIG3e_' + gene])
            for kind in ('homogeneous', 'heterogeneous'):
                g = shap_data[(shap_data.gene == gene) & (shap_data.kind == kind)]
                if g.empty:
                    continue
                own = g.set_index('cancer')[[f'shap_{o}' for o in OMICS]].T
                own.index = OMICS
                own = own.reindex(columns=paper.columns)
                fig, axs = plt.subplots(2,1, figsize=(11,4.5))
                lim = max(np.abs(paper.to_numpy()).max(), np.nanmax(np.abs(own.to_numpy())), 1e-8)
                im_paper = heat(axs[0], paper, f'{gene}: original Fig. 3e, pan-cancer feature-split experiment', True, lim)
                fig.colorbar(im_paper, ax=axs[0], fraction=.025, label='Signed SHAP')
                im = heat(axs[1], own, f'{gene}: GATrans {kind}, independently trained cancer models\nDifferent experiment; visual reference only', True, lim)
                fig.colorbar(im, ax=axs[1], fraction=.025, label='Signed SHAP')
                save(fig, out, 'gene_' + gene + '_' + kind)
                own.to_csv(out / ('gene_' + gene + '_' + kind + '.csv'))
                paper.to_csv(out / ('paper_gene_' + gene + '.csv'))
    if attention_frames:
        attention = pd.concat(attention_frames, ignore_index=True)
        for (kind,cancer,gene), data in attention.groupby(['kind','cancer','gene']):
            last = data[data.layer == data.layer.max()]
            matrix = last.pivot(index='source_slot', columns='target_slot', values='attention')
            labels = last.drop_duplicates('source_slot').sort_values('source_slot').source.tolist()
            matrix.index = matrix.columns = [f'{i}: {g}' for i,g in enumerate(labels)]
            fig, ax = plt.subplots(figsize=(6,5))
            im = heat(ax, matrix, f'{kind} {cancer}: {gene}\nChannel 0, last layer, mean over heads')
            fig.colorbar(im, ax=ax, label='Attention weight')
            save(fig, out, f'attention_{kind}_{cancer}_{gene}')
    (out / 'README.md').write_text('''# Cancer-specific figure comparison

Original values: publisher source data, DOI 10.1038/s41551-024-01312-5.
See paper_sources/manifest.json for file URLs and SHA-256 hashes.

Fig. 2b compares matching network names. Paper baseline models are published values,
not retrained models. GATrans AP and trapezoidal AUPRC are both in fold_metrics.csv.
The original ten CV seeds and tuned hyperparameters are not recovered here.

Fig. 3a compares normalized omics importance. New explanations use fold 0 only,
training-only backgrounds, correctly classified held-out positive genes, and signed
slot sums followed by mean magnitudes. The source does not fully specify the same
cohort and aggregation. Heterogeneous comparisons are extensions of the experiment.
Fig. 3b counts gene-network observations; repeated genes across networks are counted
separately. The source confidence-score samples are not supplied for cancer-specific
Fig. 3b, so only original counts are compared.

Gene heatmaps show the original pan-cancer feature-split experiment alongside our
cancer-specific models. They are different experiments, not pointwise replication.
Specific candidates and attention plots are additional exploratory model analyses.
Missing cancer/gene results stay missing. Nothing is filled from the paper.
The run_config.json and complete.json files distinguish pilot runs from ten-fold runs.
''')
    print(f'Saved figures and comparison tables to {out.resolve()}')


if __name__ == '__main__':
    main()
