import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation as mad
from pathlib import Path


rna_types_order = ['protein_coding',
'protein_coding_unused',
'retained_intron',
'nonsense_mediated_decay',
'processed_transcript',
'lincRNA',
'Mt_rRNA',
'antisense',
'Ig',
'TEC',
'sense_overlapping',
'non_stop_decay',
'sense_intronic',
'miRNA',
'Mt_tRNA',
'TCR_genes',
'other_noncoding']

rna_types_palette = {'3prime_overlapping_ncrna': '#000000',
 'Ig': '#eec39a',
 'TCR_genes': '#595652',
'other_noncoding':'#a9b394',
 'Mt_rRNA': '#ac8181',
 'Mt_tRNA': '#99e550',
 'TEC': '#3f3f74',
 'TCR': '#ac3232',
 'antisense': '#639bff',
 'lincRNA': '#f1e552',
 'macro_lncRNA': '#6abe30',
 'miRNA': '#8a6f30',
 'misc_RNA': '#847e87',
 'non_stop_decay': '#5b6ee1',
 'nonsense_mediated_decay': '#306082',
 'polymorphic_pseudogene': '#9badb7',
 'processed_transcript': '#d77bba',
 'protein_coding': '#5fcde4',
 'rRNA': '#8f974a',
 'retained_intron': '#df7126',
 'ribozyme': '#d95763',
 'sRNA': '#222034',
 'scaRNA': '#4b692f',
 'sense_intronic': '#37946e',
 'sense_overlapping': '#323c39',
 'snRNA': '#76428a',
 'snoRNA': '#696a6a',
 'vaultRNA': '#663931',
'protein_coding_unused':'#7C9A95'}

target_transcripts = pd.read_csv(
            Path(__file__)
            .resolve()
            .parent.joinpath('databases', 'coding_xena_transcripts_wo_mt_and_hist.tsv'),
        sep='\t')['ENSEMBL_ID'].to_list()

pipeline_genes = pd.read_csv(
            Path(__file__)
            .resolve()
            .parent.joinpath('databases', 'pipeline_genes.tsv'),
        sep='\t')['Gene'].to_list()

def recalculate_tpm(series):
    """   
    Recalculates TPM after gene deletion/filtration
    :param series: pandas series, genes in indeces
    :return: pandas series with recalculated TPM.
    """
    total_tpm = series.sum()
    return series.divide(total_tpm) * 1e6
    
def parse_target_id(target_id):
    """   
    Parses target_id column value (str) to extract HUGO_Gene and Transcript_Type values
    :param target_id: pandas series, name for trancript from gencode index
    :return: set of str with HUGO_Gene and Transcript_Type values 
    """
    fields = target_id.split('|')
    esembl_id = fields[0]
    hugo_gene = fields[5]       # HUGO symbol is on the 6th position
    transcript_type = fields[7]  # transcript_type is on the 8th position
    return esembl_id, hugo_gene, transcript_type
    
def intersect_genes_with_pipeline(df, return_genes=False, verbose=True):
    """
    Intersect genes from the given dataframe with the pipeline genes.

    The function uses the provided pipeline genes to identify the intersecting genes. 
    It prints the number of intersected genes and a warning if the intersection count is 
    below a certain threshold.

    :param df: DataFrame containing genes. Index of the dataframe should represent gene names (rows - genes, columns - samples).
    :param return_genes: A flag to decide whether to return the intersected genes or not. 
                         If set to True, the function will return a list of intersected genes.
    :param pipeline_genes: Set of genes to intersect with the genes in the dataframe.
                           Default is set to pipeline_genes.
    :return: If return_genes is True, returns a list of intersected genes. Otherwise, None.
    
    Example:
    --------
    >>> genes_df = pd.DataFrame({
    ...     'GeneA': [1, 2, 3],
    ...     'GeneB': [4, 5, 6],
    ...     'GeneC': [7, 8, 9]
    ... })
    >>> pipeline = {'GeneA', 'GeneD', 'GeneE'}
    >>> intersect_genes_with_pipeline(genes_df, True, pipeline)
    ['GeneA']
    """        
    intersection = set(df.index).intersection(set(pipeline_genes))
    if verbose:
        print(f'{len(intersection)} out of {len(df.index)} genes are intersected with the pipeline ({len(pipeline_genes)} genes).')
    
    if len(intersection) < 10000:
        if verbose:
            print('The amount is too low. Better check if your features are built properly with this data!')
    
    if return_genes:
        return list(intersection)



def detect_low_correlation(df, method='spearman', corr_threshold=0.6, num_corrs=2, return_index=True):
    """
    Detect and return columns (samples) from the dataframe that have low correlation with other columns.
    
    The function calculates pairwise correlations between columns of the dataframe. 
    Columns that have correlations below the specified threshold with at least a certain number of 
    other columns are considered to be low-correlated.

    :param df: DataFrame for which to compute correlations.
    :param method: Method of correlation computation. Default is 'spearman'. 
                   Other options like 'pearson' can also be used.
    :param corr_threshold: Correlation threshold below which columns are considered low-correlated.
                           Default is 0.6.
    :param num_corrs: Minimum number of other columns with which a column should have low correlation 
                      for it to be considered as low-correlated. Default is 2.
    :param return_index: A flag to decide whether to return the indices (names) of the low-correlated 
                         columns. If set to True, the function returns a list of names of low-correlated columns.
                         Default is True.
    :return: If return_index is True, returns the names of the low-correlated columns. Otherwise, None.
    
    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [2, 3, 4, 5, 6]
    ... })
    >>> detect_low_correlation(data)
    ['A', 'B']
    """

    corr = df.corr(method=method)
    less_than_threshold = corr < corr_threshold
    counts = less_than_threshold.sum()
    selected_cols = counts[counts >= num_corrs] 
    avg_corr = corr[selected_cols.index].values.mean()

    if len(selected_cols) > 0:
        print("Average correlation of low-correlated sample:", avg_corr)
        print('Low-correlated samples are:')
        print('\n'.join(selected_cols.index))
    else:
        print("All samples are well-correlated")
    
    if return_index:
        return selected_cols.index 


def iterative_pca_outliers(df, return_labels=True):
    """
    Detects and returns outliers from a dataframe using iterative PCA and the Median Absolute Deviation (MAD) method.
    
    The function performs PCA to reduce the data to two components. It then identifies outliers 
    iteratively based on the deviation from the median in the PCA space using the MAD method. 
    After each iteration, identified outliers are removed, and PCA is repeated until no more outliers are detected.

    :param df: DataFrame from which to detect outliers  (rows - samples, columns - genes). Outliers are detected for the rows. Dataframe should have no NaNs.
    :param return_labels: A flag to decide whether to return the labels (names) of the detected outliers.
                          If set to True, the function returns a list of names of outliers.
                          Default is True.
    :return: If return_labels is True, returns the names of the detected outliers. Otherwise, None.
    
    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 50],
    ...     'B': [1, 2, 3, 4, 60],
    ...     'C': [1, 2, 3, 4, 70]
    ... })
    >>> iterative_pca_outliers(data)
    ['A', 'B', 'C']
    """

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
    return labels

def check_log_scale(expressions, return_df=True, return_bool=False):
    """
    Detects log scaling by max mean value, return scaled df if seems to be raw (max mean value > 20 TPM)
    :param expressions: pandas.DataFrame - RNAseq expression data with TPM
    :param return_df: bool, switcher to whether return dataframe or not
    :param return_bool: bool, will return bool value if switched on: returns False if not log-scaled; True if log-scaled.
    :return: if seems to be non-scaled, scaled expressions, otherwise same dataframe
    """
    if expressions.mean().max()>20:
        if return_bool:
            return False
        else:
            print('Seems to be unscaled, scaling...')
            expressions = np.log2(expressions+1)
            print('Scaled')
        
    else:
        if return_bool:
            return True
        else:
            print('Seems to be log-scaled')
        
    if return_df:
        return expressions

def percent_zero_columns_for_intersected_genes(df):
    """
    Checks the percentage of intersected genes with the pipeline that have all-zero columns.
    
    :param df: DataFrame to check. Genes should be in rows.
    :return: Percentage of intersected genes with all-zero columns.
    
    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'A': [0, 2, 0, 0, 5],
    ...     'B': [0, 0, 8, 0, 10],
    ...     'C': [0, 12, 13, 0, 15]
    ... }, index=['gene1', 'gene2', 'gene3', 'gene4', 'gene5'])
    >>> genes = ['gene1', 'gene2', 'gene3', 'gene6']
    >>> percent_zero_columns_for_intersected_genes(data, genes)
    33.33
    """
    intersection = intersect_genes_with_pipeline(df, return_genes=True, verbose=False)
    df = df.loc[intersection]
    all_zero_genes = df.index[df.T.sum()==0]
    zero_genes_count = len(all_zero_genes)
    percent_zero = round((zero_genes_count / len(intersection)) * 100, 2)
    print(f"{percent_zero}%({zero_genes_count}) of intersected genes are all equal to 0, if all values are non-negative.")


def check_negative_values(df):
    """
    Checks the provided dataframe for any negative values and returns a warning if found.
    
    :param df: DataFrame to check for negative values.
    :return: A warning message if any negative value is found in the dataframe. 
             Otherwise, None.
             
    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'A': [1, -2, 3, 4, 5],
    ...     'B': [6, 7, 8, 9, 10],
    ...     'C': [11, 12, 13, 14, 15]
    ... })
    >>> check_negative_values(data)
    Warning: The dataset contains negative values!
    """

    if (df < 0).any().any():
        print("Warning: The dataset contains negative values!")
    else:
        print("The dataset has no negative values.")