import warnings
import community  # louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from functions.utils import GeneSet, mean_confidence_interval
from functions.plotting import axis_matras, bot_bar_plot, lin_colors

import matplotlib

def query_genes_by_symbol(genes, verbose=False):
    """

    :param genes:
    :param verbose:
    :return:
    """
    import mygene

    mg = mygene.MyGeneInfo()
    q = mg.querymany(
        genes,
        species='human',
        as_dataframe=True,
        verbose=verbose,
        df_index=True,
        scopes=["symbol"],
        fields="all",
    )
    try:
        q.dropna(subset=['HGNC'], inplace=True)
        q.dropna(subset=['type_of_gene'], inplace=True)
        q.dropna(subset=['map_location'], inplace=True)
    except Exception:
        warnings.warn('Output lacks map_location or type_of_gene')

    return q

def update_gene_names(genes_old, genes_cur, verbose=False, using_genestorage=False):
    """
    Takes a set of gene names genes_old and matches it with genes_cur.
    For all not found tries to match with known aliases using mygene or GeneStorage.
    All not matched are returned as is.
    Returns a dict with matching rule. No duplicates will be in output
    :param genes_old:
    :param genes_cur:
    :param verbose:
    :param using_genestorage:
    :return:
    """
    c_genes = set(genes_cur)
    old_genes = set(genes_old)

    missing = set()

    common_genes = c_genes.intersection(old_genes)
    if verbose:
        print('Matched: {}'.format(len(common_genes)))

    converting_genes = old_genes.difference(c_genes)
    rest_genes = c_genes.difference(old_genes)
    match_rule = {cg: cg for cg in common_genes}

    if len(converting_genes):

        if verbose:
            print(
                'Trying to find new names for {} genes in {} known'.format(
                    len(converting_genes), len(rest_genes)
                )
            )

        qr = query_genes_by_symbol(list(converting_genes), verbose=verbose)
        if hasattr(qr, 'alias'):
            cg_ann = qr.alias.dropna()
        else:
            cg_ann = pd.DataFrame()

        for cg in converting_genes:
            if cg in cg_ann.index:
                if (isinstance(cg_ann.loc[cg], list)) | (
                    isinstance(cg_ann.loc[cg], pd.core.series.Series)
                ):
                    al_set = set(cg_ann[cg])
                else:
                    al_set = set([cg_ann.loc[cg]])

                hits = al_set.intersection(rest_genes)
                if len(hits) == 1:
                    match_rule[cg] = list(hits)[0]
                    rest_genes.remove(match_rule[cg])
                elif len(hits) > 1:
                    warnings.warn('{} hits for gene {}'.format(len(hits), cg))
                    match_rule[cg] = list(hits)[0]
                    rest_genes.remove(match_rule[cg])
                else:
                    missing.add(cg)
                    match_rule[cg] = cg
            else:
                missing.add(cg)
                match_rule[cg] = cg
        if verbose and len(missing):
            print('{} genes were not converted'.format(len(missing)))
    return match_rule

def run_progeny(expression_df, sync_gene_names=True, prog_coeffs=None, **kwargs):
    """
    Runs PROGENy pathway scoring on provided expressions dataframe in python
    :param expression_df: pd.DataFrame; rows - Hugo Gene symbols, columns - samples
    :param prog_coeffs: pd.DataFrame, progeny_genes_coefficients; index - HUGO gene symbols, columns -
                        ['pathway', 'coefficient']
    :returns progeny pathway scores dataframe
    """
    if prog_coeffs is None:
        prog_coeffs = pd.read_csv(
            Path(__file__)
            .resolve()
            .parent.joinpath('databases', 'progeny_genes_coefficients.tsv'),
            index_col=None,sep='\t'
        )
    if sync_gene_names:
        matching_genes = update_gene_names(
            genes_old=prog_coeffs['hugo_symbol'],
            genes_cur=expression_df.index,
            **kwargs
        )

        prog_coeffs = prog_coeffs.assign(
            hugo_symbol=prog_coeffs.hugo_symbol.map(matching_genes)
        )

    coeffs = pd.pivot_table(
        prog_coeffs,
        index=['hugo_symbol'],
        columns=['pathway'],
        values='coefficient',
        aggfunc=sum,
        fill_value=0,
    )

    return expression_df.reindex(coeffs.index, fill_value=0).T.dot(coeffs).T

def gen_graph(similarity_matrix, threshold=0.8):
    """
    Generates a graph from the similarity_matrix (square dataframe). Each sample is a node, similarity - edge weight.
    Edges with weight lower than the threshold are ignored.
    Only nodes with at least 1 edge with weight above the threshold will be present in the final graph
    :param similarity_matrix:
    :param threshold:
    :return:
    """
    G = nx.Graph()
    for col_n in similarity_matrix:
        col = similarity_matrix[col_n].drop(col_n)
        mtr = col[col > threshold]
        for row_n, val in list(mtr.to_dict().items()):
            G.add_edge(col_n, row_n, weight=round(val, 2))
    return G


def louvain_community(correlation_matrix, threshold=1.4, resolution=1, random_state=100, **kwargs):
    """
    Generates a graph from a correlation_matrix with weighted edges (weight<threshold excluded).
    Then detects communities using louvain algorithm (https://github.com/taynaud/python-louvain)
    :param correlation_matrix:
    :param threshold:
    :param resolution:
    :param random_state:
    :param kwargs:
    :return:
    """
    G = gen_graph(correlation_matrix, threshold)
    if len(G.edges()) > 10000000:
        warnings.warn("Too many edges will result in huge computational time")
    return pd.Series(community.best_partition(G, resolution=resolution,
                                              random_state=random_state, **kwargs)) + 1


def dense_clustering(data, threshold=0.4, name='MFP', method='louvain', **kwargs):
    """
    Generates a graph from the table features(cols)*samples(rows).
    Then performs community detection using a selected method leiden|louvain
    :param method:
    :param data:
    :param threshold:
    :param name:
    :return:
    """
    if method == 'louvain':  # others could be implemented like Leiden
        partition = louvain_community(data.T.corr() + 1, threshold + 1, **kwargs)
    else:
        raise Exception('Unknown method')

    return partition.rename(name)


def clustering_profile_metrics(data, threshold_mm=(0.3, 0.65), step=0.01, method='louvain'):
    """
    Iterates threshold in threshold_mm area with step. Calculates cluster separation metrics on each threshold.
    Returns a pd.DataFrame with the metrics
    :param data:
    :param threshold_mm:
    :param step:
    :param method:
    :return:
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    cluster_metrics = {}

    for tr in tqdm(np.round(np.arange(threshold_mm[0], threshold_mm[1], step), 3)):
        clusters_comb = dense_clustering(data, threshold=tr, method=method)
        cluster_metrics[tr] = {
            'ch': calinski_harabasz_score(data.loc[clusters_comb.index], clusters_comb),
            'db': davies_bouldin_score(data.loc[clusters_comb.index], clusters_comb),
            'sc': silhouette_score(data.loc[clusters_comb.index], clusters_comb),
            'N': len(clusters_comb.unique()),
            'perc': clusters_comb,
        }

    return pd.DataFrame(cluster_metrics).T


def clustering_profile_metrics_plot(cluster_metrics, num_clusters_ylim_max=7):
    """
    Plots a dataframe from clustering_profile_metrics
    :param cluster_metrics:
    :param num_clusters_ylim_max:
    :return: axis array
    """
    # necessary for correct x axis sharing
    cluster_metrics.index = [str(x) for x in cluster_metrics.index]

    plots_ratios = [3, 3, 3, 1, 2]
    fig, axs = plt.subplots(len(plots_ratios), 1, figsize=(8, np.sum(plots_ratios)),
                            gridspec_kw={'height_ratios': plots_ratios}, sharex=True)
    for ax in axs:
        ax.tick_params(axis='x', which='minor', length=0)
    af = axs.flat

    ax = cluster_metrics.db.plot(ax=next(af), label='Davies Bouldin', color='#E63D06')
    ax.legend()

    ax = cluster_metrics.ch.plot(ax=next(af), label='Calinski Harabasz', color='#E63D06')
    ax.legend()

    ax = cluster_metrics.sc.plot(ax=next(af), label='Silhouette score', color='#E63D06')
    ax.legend()

    ax = cluster_metrics.N.plot(kind='line', ax=next(af), label='# clusters', color='#000000')
    ax.set_ylim(0, num_clusters_ylim_max)
    ax.legend()

    # display percentage for 10 clusters max
    clusters_perc = pd.DataFrame([x.value_counts() for x in cluster_metrics.perc],
                                 index=cluster_metrics.index).iloc[:, :10]

    clusters_perc.plot(kind='bar', stached=True, ax=next(af), offset=.5)

    ax.set_xticks(ax.get_xticks() - .5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_ylabel('Cluster %')
    return ax


def clustering_select_best_tr(data, n_clusters=4, threshold_mm=(0.3, 0.6),
                              step=0.025, method='leiden', num_clusters_ylim_max=7, plot=True):
    """
    Selects the best threshold for n_clusters separation using dense_clustering with selected method
        from threshold_mm with a particular step
    :param data: dataframe with processes (rows - samples, columns - signatures)
    :param n_clusters: desired number of clusters
    :param threshold_mm: range of thresholds
    :param step: step to go through range of thresholds
    :param method: clustering method
    :param num_clusters_ylim_max: set y_lim for plot with number of clusters
    :param plot: whether to plot all matrix
    :return: the threshold to get n_clusters
    """
    cl_scs = clustering_profile_metrics(data, threshold_mm=threshold_mm, step=step, method=method)

    if plot:
        clustering_profile_metrics_plot(cl_scs, num_clusters_ylim_max)
        plt.show()

    cl_scs_filtered = cl_scs[cl_scs.N == n_clusters]

    if not len(cl_scs_filtered):
        raise Exception('No partition with n_clusters = {}'.format(n_clusters))

    cl_scs_filtered.sc += 1 - cl_scs_filtered.sc.min()
    return (cl_scs_filtered.ch / cl_scs_filtered.db / cl_scs_filtered.sc).sort_values().index[-1]



def read_gene_sets(gmt_file):
    """
    Return dict {geneset_name : GeneSet object}

    :param gmt_file: str, path to .gmt file
    :return: dict
    """
    gene_sets = {}
    with open(gmt_file) as handle:
        for line in handle:
            items = line.strip().split('\t')
            name = items[0].strip()
            description = items[1].strip()
            genes = set([gene.strip() for gene in items[2:]])
            gene_sets[name] = GeneSet(name, description, genes)

    return gene_sets

                                  
def gmt_genes_alt_names(gmt, genes, verbose=False, report_missing=False, **kwargs):
    """
    Updates gmt with genes aliases found in genes
    :param gmt: read_gene_sets() function result
    :param genes: list/set of genes available for current platform
    :param verbose: if True then prints all mismatched genes
    :param report_missing: report additional set with genes failed to convert
    :return: read_gene_sets() return like with updated or removed genes
    """
    s_genes = set(genes)
    alt_gmt = {}
    gmt_genes = set()

    for geneset in gmt:
        gmt_genes.update(gmt[geneset].genes)

    match_rule = update_gene_names(
        genes_old=gmt_genes, genes_cur=s_genes, verbose=verbose, **kwargs
    )

    missing_genes = set()
    for geneset in gmt:
        new_set = set()
        for gene in gmt[geneset].genes:
            if match_rule[gene] != gene or gene in s_genes:
                new_set.add(match_rule[gene])
            else:
                missing_genes.add(gene)
        alt_gmt[geneset] = GeneSet(
            name=gmt[geneset].name, descr=gmt[geneset].descr, genes=new_set
        )

    if report_missing:
        return alt_gmt, missing_genes
    return alt_gmt


def SWISS(data, groups, group_mean=True):
    """
    Calculates SWISS score: Standardized WithIn Class Sum of Squares (https://doi.org/10.1371/journal.pone.0009905). Score is a ratio of within-class residual (value minus class average) sum of squares, or WISS, and total residual (value minus global average) sum of squares, or SST. The lower the score (WISS/SST) the better the clustering.
    :param data: dataframe of features for clustering, indeces = patients, columns = genes
    :param groups: pd.Series, indeces = patients, labels = clustering labels
    :param group_mean: bool, if False - use global average for the SST calculations, i.e. data mean. If True = use mean of group means as a global average for the SST calculations, i.e. scale the mean if the cluster size seriously differs.  Default is True.
    :return: float, SWISS score.
    """
    d1, n = data.shape
    k = groups.max()

    # Вычисление средних значений групп
    gmeanA = np.zeros((d1, k))
    for i in range(1, k + 1):
        group_data = data[:, groups == i]
        gmeanA[:, i - 1] = group_data.mean(axis=1)

    # Вычисление общего среднего
    omeanA = np.zeros((d1, 1))
    if not group_mean:
        omeanA = data.mean(axis=1).reshape(-1, 1)
    else:
        omeanA = gmeanA.mean(axis=1).reshape(-1, 1)

    # Вычисление матриц расстояний
    distA = np.zeros((2, n))
    for i in range(n):
        g = groups[i]
        distA[0, i] = np.sum((data[:, i] - gmeanA[:, g - 1]) ** 2)
        distA[1, i] = np.sum((data[:, i] - omeanA.flatten()) ** 2)

    # Расчет отношения A и B
    RatioA = np.sum(distA[0, :]) / np.sum(distA[1, :])

    return RatioA


def clustering_profile_metrics_mean_plot(cluster_metrics, cluster_of_interest=None, select_better=False):
    """
    Plots a dataframe from clustering_profile_metrics with mean and 95% CI as shaded area.
    :param cluster_metrics: Nested dictionary with cluster metrics.
    :param cluster_of_interest: Number of the cluster of interest. If None, select the best cluster based on 'product' metric.
    :param select_better: If True and cluster_of_interest is None, select the cluster with the highest 'product' metric.
    :return: axis array
    """
    metrics = {'db':'Davies-Bouldin', 
               'ch':'Calinski-Harabasz', 
               'sc':'Silouhette', 
               'sw':'SWISS Score',
               'spar_w':'Sparsity (Basis matrix)',
               'spar_h':'Sparsity (Feature matrix)',
               'coph_cor':'Cophenetic correlation',
               'evar':'Explained variance',
               'product':'Product of all scores,\nexcept Sparsity'} 
    
    # Convert the nested dictionary to a DataFrame
    df_list = []
    percents = {}
    for n_clusters, rounds in cluster_metrics.items():
        percents[n_clusters] = pd.concat([cluster_metrics[n_clusters][i]['perc'].value_counts() for i in cluster_metrics[n_clusters].keys()],axis=1).T.mean().sort_values()
        for round_num, scores in rounds.items():
            scores['n_clusters'] = n_clusters
            df_list.append(scores)
            
    df = pd.DataFrame(df_list)

    if select_better and cluster_of_interest is None:
        cluster_of_interest = df.groupby('n_clusters')['product'].mean().idxmax()

    df = pd.DataFrame(df_list)
    # Initialize plot
    af = axis_matras([1]*(len(metrics.keys()))+[1, 1.5], title=None, x_len=10)

    # Plot each metric with CI as shaded area
    for metric in list(metrics.keys()):
        metric_data = df.groupby('n_clusters')[metric].apply(list).reset_index()
        metric_data.index = [i for i in metric_data.n_clusters]
        metric_data['mean'], metric_data['ci_lower'], metric_data['ci_upper'] = zip(*metric_data[metric].apply(mean_confidence_interval))

        ax = next(af)
        ax.plot([str(i) for i in metric_data.n_clusters], metric_data['mean'], label=metrics[metric], color='#E63D06')
        ax.fill_between([str(i) for i in metric_data.n_clusters], metric_data['ci_lower'], metric_data['ci_upper'], color='#E63D06', alpha=0.2)
        ax.legend()
        ax.axvline(x=cluster_of_interest, color='black', linestyle='--', linewidth=1)

    ax = pd.Series(cluster_metrics.keys()).plot(
        kind='line', ax=next(af), label='# clusters', color='#000000'
    )
    ax.legend()
    ax.axvline(x=cluster_of_interest, color='black', linestyle='--', linewidth=1)

    clusters_perc = pd.DataFrame(percents).T

    ax = bot_bar_plot(
        clusters_perc,percent=True,
        ax=next(af),
        legend=False,
        offset=0.5,
        palette=lin_colors(pd.Series(clusters_perc.columns), cmap=matplotlib.cm.tab20c),
    )
    ax.axvline(x=cluster_of_interest, color='black', linestyle='--', linewidth=1)

    ax.set_xticks(ax.get_xticks() - 0.5)
    ax.set_xticklabels(cluster_metrics.keys(), rotation=90)

    ax.set_ylabel('Cluster %')
    return af