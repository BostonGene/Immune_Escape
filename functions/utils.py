import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def star_pvalue(pvalue, lev1=0.05, lev2=0.01, lev3=0.001):
    """
    Return star notation for p value
    :param pvalue: float
    :param lev1: float, '*' threshold
    :param lev2: float, '**' threshold
    :param lev3: float, '***' threshold
    :return: str
    """
    if pvalue < lev3:
        return '***'
    if pvalue < lev2:
        return '**'
    if pvalue < lev1:
        return '*'
    return '-'

def round_to_1(x):
    """
    Round "x" to first significant digit
    :param x: float
    :return: float
    """
    from math import floor, log10

    return round(x, -int(floor(log10(abs(x)))))

def get_pvalue_string(p, p_digits=3, stars=False, prefix='p-value'):
    """
    Return string with p-value, rounded to p_digits
    :param stars: Display pvalue as stars
    :param p: float, p-value
    :param p_digits: int, default 4, number of digits to round p-value
    :param prefix:
    :return: str, p-value info string
    """
    significant_pvalue = 10**-p_digits
    if stars:
        pvalue_str = star_pvalue(p, lev3=10**-p_digits)
    else:
        if p < significant_pvalue:
            if len(prefix):
                prefix += ' < '
            pvalue_str = f'{prefix}{significant_pvalue}'
        else:
            if len(prefix):
                prefix += ' = '
            if p < 0.00001:
                pvalue_str = f'{prefix}{round_to_1(p):.0e}'
            else:
                pvalue_str = f'{prefix}{round_to_1(p)}'
    return pvalue_str
    
def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate mean and confidence interval.
    :param data: pandas series to evaluate
    :param confidence: float, confidence level percentage (default 95%)
    :return: tuple of mean, lower CI, upper CI
    """
    from scipy import stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class GeneSet(object):
    def __init__(self, name, descr, genes):
        self.name = name
        self.descr = descr
        self.genes = set(genes)
        self.genes_ordered = list(genes)

    def __str__(self):
        return '{}\t{}\t{}'.format(self.name, self.descr, '\t'.join(self.genes))
        
def scale_series(series,feature_range=(0, 1)):
    name = series.name
    scaler = MinMaxScaler(feature_range=feature_range)
    series_2d = series.values.reshape(-1, 1)
    scaled_series_2d = scaler.fit_transform(series_2d)
    scaled_series = pd.Series(scaled_series_2d.flatten(), index=series.index)
    scaled_series.name =name
    return scaled_series

    
def df_fisher_chi2(clusters = pd.Series, response=pd.Series, R=False, NR=True):
    import pandas as pd
    from scipy.stats import fisher_exact, chi2_contingency
    from statsmodels.stats.multitest import multipletests
    df=pd.crosstab(clusters,response)
    df.insert(0,'Fisher_pv', 1)
    df.insert(1,'Chi2_pv', 1)

    for  i in df.index:
        nr, r = df[R].loc[i], df[NR].loc[i]
        nrj = df[R].sum() - nr
        rj = df[NR].sum() - r
        oddsratio, pvalue = fisher_exact([[nr, r],[nrj, rj]])  
        if pvalue > 1:
            pvalue = 1
        df.at[i,'Fisher_pv'] = pvalue

    for  i in df.index:
        nr, r = df[R].loc[i], df[NR].loc[i]
        nrj = df[R].sum() - nr
        rj = df[NR].sum() - r
        chi, pvalue, dof, exp = chi2_contingency([[r, nr],[rj, nrj]])  
        if pvalue > 1:
            pvalue = 1
        df.at[i, 'Chi2_pv'] = pvalue
    _, df['Fisher_pv'],_,_ = multipletests(df['Fisher_pv'],method='fdr_bh')
    _, df['Chi2_pv'],_,_ = multipletests(df['Chi2_pv'],method='fdr_bh')
    return df
    
def sort_by_terms_order(
    data: pd.Series, t_order: list, vector: pd.Series = None
) -> np.ndarray:
    """
    Sort "data" into blocks with values from "t_order". If "vector" is provided, sort each block by corresponding
    values in "vector"
    :param data: pd.Series
    :param t_order: list, values for blocks to sort "data" into
    :param vector: pd.Series, same index as data, which values to sort each block by
    :return: np.array, 1 dimensional
    """

    x = []
    for term in t_order:
        indices = data[data == term].index

        if len(indices):
            if vector is not None:
                x.append(vector.reindex(indices).dropna().sort_values().index)
            else:
                x.append(indices)

    return np.concatenate(x)

def ssgsea_score(ranks, genes, use_old_formula = False):
    """
    Calculates single sample GSEA score based on vector of gene expression ranks.
    Only overlapping genes will be analyzed.
    The original article describing the ssGSEA formula: https://doi.org/10.1038/nature08460.
    We use adapted fast function. Result is the same as in analogous packages (like GSVA).

    Note: formula was updated in November 2023.
    Please visit Wiki-page for more details: https://bostongene.atlassian.net/wiki/spaces/BIT/pages/3427991553/ssGSEA

    :param ranks: DataFrame with gene expression ranks; samples in columns and genes in rows
    :param genes: list or set, genes of interest
    :param use_old_formula: (Default: False) if true calculates old ssGSEA-BostonGene Formula
    :return: Series with ssGSEA scores for samples
    """

    # Finding common_genes
    # Note: List is needed here because pandas can not do .loc with sets
    common_genes = list(set(genes).intersection(set(ranks.index)))

    # If not intersections were found
    if not common_genes:
        return pd.Series([0.0] * len(ranks.columns), index=ranks.columns)

    # Ranks of genes inside signature
    sranks = ranks.loc[common_genes]

    if use_old_formula:
        return (sranks ** 1.25).sum() / (sranks ** 0.25).sum() - (len(ranks.index) - len(common_genes) + 1) / 2

    return (sranks ** 1.25).sum() / (sranks ** 0.25).sum() - (len(ranks.index) + 1) / 2


def ssgsea_formula(data, gene_sets, rank_method='max'):
    """
    Return DataFrame with ssgsea scores
    Only overlapping genes will be analyzed

    :param data: pd.DataFrame, DataFrame with samples in columns and variables in rows
    :param gene_sets: dict, keys - processes, values - bioreactor.gsea.GeneSet
    :param rank_method: str, 'min' or 'max'.
    :return: pd.DataFrame, ssgsea scores, index - genesets, columns - patients
    """

    ranks = data.T.rank(method=rank_method, na_option='bottom')

    return pd.DataFrame({gs_name: ssgsea_score(ranks, gene_sets[gs_name].genes)
                         for gs_name in list(gene_sets.keys())})


def median_scale(data, clip=None, c=1., exclude=None, axis=0):
    """
    Scale using median and mad over any axis (over columns by default).
    Removes the median and scales the data according to the median absolute deviation.

    To calculate Median Absolute Deviation (MAD) - function "mad" from statsmodels.robust.scale is used
    with arguments "c" equals to 1, hence no normalization is performed on MAD
    [please see: https://www.statsmodels.org/stable/generated/statsmodels.robust.scale.mad.html]

    :param data: pd.DataFrame of pd.Series
    :param clip: float, symmetrically clips the scaled data to the value
    :param c: float, coefficient of normalization used in calculation of MAD
    :param exclude: pd.Series, samples to exclude while calculating median and mad
    :param axis: int, default=0, axis to be applied on: if 0, scale over columns, otherwise (if 1) scale over rows

    :return: pd.DataFrame
    """
    from statsmodels.robust.scale import mad

    if exclude is not None:
        data_filtered = data.reindex(data.index & exclude[~exclude].index)
    else:
        data_filtered = data

    median = 1.0 * data_filtered.median(axis=axis)

    if isinstance(data, pd.Series):
        madv = 1.0 * mad(data_filtered.dropna(), c=c)
        c_data = data.sub(median).div(madv)
    else:
        inv_axis = (axis + 1) % 2  # Sub and div are performed by the other axis
        madv = 1.0 * data_filtered.apply(lambda x: mad(x.dropna(), c=c), axis=axis)
        c_data = data.sub(median, axis=inv_axis).div(madv, axis=inv_axis)

    if clip is not None:
        return c_data.clip(-clip, clip)
    return c_data


def read_dataset(file, sep='\t', header=0, index_col=0, comment=None):
    return pd.read_csv(file, sep=sep, header=header, index_col=index_col,
                       na_values=['Na', 'NA', 'NAN'], comment=comment)


def item_series(item, indexed=None):
    """
    Creates a series filled with item with indexes from indexed (if Series-like) or numerical indexes (size=indexed)
    :param item: value for filling
    :param indexed:
    :return:
    """
    if indexed is not None:
        if hasattr(indexed, 'index'):
            return pd.Series([item] * len(indexed), index=indexed.index)
        elif type(indexed) is int and indexed > 0:
            return pd.Series([item] * indexed, index=np.arange(indexed))
    return pd.Series()


def to_common_samples(df_list=()):
    """
    Accepts a list of dataframes. Returns all dataframes with only intersecting indexes
    :param df_list: list of pd.DataFrame
    :return: pd.DataFrame
    """
    cs = set(df_list[0].index)
    for i in range(1, len(df_list)):
        cs = cs.intersection(df_list[i].index)

    if len(cs) < 1:
        warnings.warn('No common samples!')
    return [df_list[i].loc[list(cs)] for i in range(len(df_list))]


def cut_clustermap_tree(g, n_clusters=2, by_cols=True, name='Clusters'):
    """
    Cut clustermap into desired number of clusters. See scipy.cluster.hierarchy.cut_tree documentation.
    :param g:
    :param n_clusters:
    :param by_cols:
    :param name:
    :return: pd.Series
    """
    from scipy.cluster.hierarchy import cut_tree
    if by_cols:
        link = g.dendrogram_col.linkage
        index = g.data.columns
    else:
        link = g.dendrogram_row.linkage
        index = g.data.index

    return pd.Series(cut_tree(link, n_clusters=n_clusters)[:, 0], index=index, name=name) + 1


def pivot_vectors(vec1, vec2, na_label_1=None, na_label_2=None):
    """
    Aggregates 2 vectors into a table with amount of pairs (vec1.x, vec2.y) in a cell
    Both series must have same index.
    Else different indexes values will be counted in a_label_1/na_label_2 columns if specified or ignored
    :param vec1: pd.Series
    :param vec2: pd.Series
    :param na_label_1: How to name NA column
    :param na_label_2: How to name NA row
    :return: pivot table
    """

    name1 = str(vec1.name)
    if vec1.name is None:
        name1 = 'V1'

    name2 = str(vec2.name)
    if vec2.name is None:
        name2 = 'V2'

    if name1 == name2:
        name1 += '_1'
        name2 += '_2'

    sub_df = pd.DataFrame({name1: vec1,
                           name2: vec2})
    # FillNAs
    fill_dict = {}
    if na_label_1 is not None:
        fill_dict[name1] = na_label_1
    if na_label_2 is not None:
        fill_dict[name2] = na_label_2
    sub_df.fillna(value=fill_dict, inplace=True)

    sub_df = sub_df.assign(N=item_series(1, sub_df))

    return pd.pivot_table(data=sub_df, columns=name1,
                          index=name2, values='N', aggfunc=sum).fillna(0).astype(int)

def iterative_pca_outliers(df, return_labels=True):
    from sklearn.decomposition import PCA
    from scipy.stats import median_abs_deviation as mad
    pca = PCA(n_components=2)
    continue_search = True
    labels = []
    data = df.values
    index = df.index
    while continue_search:
        transformed_data = pca.fit_transform(data)
        medians = np.median(transformed_data, axis=0)
        mads = mad(transformed_data, scale=1/1.4826, axis=0)
        a = np.abs((transformed_data - medians) / mads)
        outliers_idx = np.any(a > 6, axis=1)
        if np.any(outliers_idx):
            for idx in index[outliers_idx]:
                print(f'{idx} is detected as an outlier')
                labels.append(idx)
            data = data[~outliers_idx]
            index = index[~outliers_idx]
        else:
            continue_search = False
            print('There are no outliers')
    if return_labels:
        return labels

