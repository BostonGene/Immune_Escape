import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.robust.scale import mad
from scipy.stats import sem
from scipy.stats.distributions import rv_continuous
from math import floor, log10

def predict_batch(ser, X, batch,k=20):
    """
    Calculates euclidean distances for the sample ser in the set space X, then returns the most frequent identified batch from k nearest neighbors. 
    In case there are 2 equally frequent batches, chooses the first one in the list.
    :param ser: pandas.Series, indeces - features, values without NaN
    :param X: pandas.DataFrame, indeces - samples, columns - features, identical to ser.index; values without NaN
    :param batch: pandas.Series, indeces - samples, same as for X.index, values - str values with batch name
    :param k: int, number of k nearest neighbors from which to choose the most frequent batch
    :return: str, the most frequent batch
    """
    distances = np.sqrt(((X - ser) ** 2).sum(axis=1))

    nearest_indices = distances.nsmallest(k).index

    nearest_batches = batch.loc[nearest_indices]

    batch_counts = nearest_batches.value_counts()

    max_count = batch_counts.max()
    most_frequent_batches = batch_counts[batch_counts == max_count].index

    for batch_value in nearest_batches:
        if batch_value in most_frequent_batches:
            most_frequent_batch = batch_value
            break

    return most_frequent_batch

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
    
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * rv_continuous.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class GeneSet(object):
    def __init__(self, name, descr, genes):
        self.name = name
        self.descr = descr
        self.genes = set(genes)
        self.genes_ordered = list(genes)

    def __str__(self):
        return '{}\t{}\t{}'.format(self.name, self.descr, '\t'.join(self.genes))
        
def scale_series(series, feature_range=(0, 1)):
    min_val = series.min()
    max_val = series.max()
    range_min, range_max = feature_range
    scaled_data = (series - min_val) * (range_max - range_min) / (max_val - min_val) + range_min
    scaled_series = pd.Series(scaled_data, index=series.index)
    scaled_series.name = series.name
    return scaled_series


    
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


def read_dataset(file, sep='\t', header=0, index_col=0, comment=None, usecols=None):
    return pd.read_csv(file, sep=sep, header=header, index_col=index_col,
                       na_values=['Na', 'NA', 'NAN'], comment=comment, usecols=usecols)


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

