import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation as mad
import os
from tqdm import tqdm_notebook, tqdm
import glob
from pathlib import Path
import tarfile
import subprocess
import sys
import shutil
import zipfile

lgrey_color = '#AAAAAA'

rna_types_palette = {'3prime_overlapping_ncrna': '#000000',
 'Ig': '#eec39a',
 'IG_V_gene': '#595652',
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

cells_p = {'B_cells': '#004283',
 'Plasma_B_cells': '#0054A8',
 'Non_plasma_B_cells': '#0066CC',
 'Mature_B_cells': '#3889DB',
 'Naive_B_cells': '#78B0E9',
 'T_cells': '#285A51',
 'CD8_T_cells': '#31685E',
 'CD8_T_cells_PD1_high': '#3C776C',
 'CD8_T_cells_PD1_low': '#3C776C',
 'CD4_T_cells': '#61A197',
 'Th': '#70B0A5',
 'Th1_cells': '#7FBEB3',
 'Th2_cells': '#8FCCC2',
 'Th17_cells': '#9DD4C9',
 'Naive_T_helpers': '#ACDCD3',
 'Tregs': '#CBEBE6',
 'NK_cells': '#6181A1',
 'Cytotoxic_NK_cells': '#7F9EBE',
 'Regulatory_NK_cells': '#9DB8D4',
 'Myeloid_cells': '#8C0021',
 'Monocytes': '#6A3C77',
 'Macrophages': '#865494',
 'Macrophages_M1': '#A370B0',
 'Macrophages_M2': '#BF8FCC',
 'Microglia': '#6B4F73',
 'MDSC': '#9F86A6',
 'Granulocytes': '#D93158',
 'Eosinophils': '#B7002B',
 'Neutrophils': '#EC849C',
 'Basophils': '#854855',
 'Mast_cells': '#B0707D',
 'Dendritic_cells': '#50285B',
 'Endothelium': '#DCB7AC',
 'Vascular_endothelium_cells': '#DCB7AC',
 'Lymphatic_endothelium_cells': '#998078',
 'Stromal_cells': '#CC7A00',
 'Fibroblasts': '#FF9500',
 'iCAF': '#FFB341',
 'myCAF': '#FFCD83',
 'Follicular_dendritic_cells': '#D2871E',
 'Adypocytes': '#ECDAA7',
 'Fibroblastic_reticular_cells': '#995B00',
 'Other': '#C2C1C7',
 'Epithelial_cells': '#DFD3CF',
 'Muscles': '#DF714B',
 'Bones': '#96A4B3'}

cells_o = ['NK_cells',
 'CD4_T_cells',
 'CD8_T_cells',
 'B_cells',
 'Monocytes',
 'Macrophages',
 'Neutrophils',
 'Fibroblasts',
 'Endothelium',
 'Other']

def fastqc_unzip(
        infile: str,
        edir: str
):
    """
    Unzip archive to the directory.

    :param infile: zip file
    :param edir: target directory
    """
    print(f'Unzipping {infile} to {edir}...')
    # Unzip files and put to another folder
    with zipfile.ZipFile(infile, "r") as zip_ref:
        random_folder_name = zip_ref.namelist()[0]
        zip_ref.extractall(edir)
    return random_folder_name


def fastqc_clean(edir: str):
    """
    Clean the directory.

    :param edir: directory to clean
    """
    print(f'Cleaning directory {edir}...')
    return shutil.rmtree(Path(edir))


def create_fastqc_dirs(
        cohort_id: str,
        patient: str,
        fastqc_zip: str,
        cohorts_folder: str,
        files_type: str = 'raw'
):
    """
    Create necessary working directories and return paths to save data.

    :param cohort_id: name of cohort (e.g. "Overman_PAAD")
    :param patient: patient id
    :param fastqc_zip: name of archived raw fastqc file
    :param cohorts_folder: directory where you want to store unpacked and processed files e.g. "/home/sonya/qc"
    :param files_type: part of filename before extension e.g. "*.fastqc_<files_type>.zip"
    :return: names of input raw data and filenames where to store postprocessed data for fastqc_cohort function
    """
    # Getting patient id
    sample_id = fastqc_zip.replace(f'.fastqc_{files_type}.zip', '')

    # It seems useless now
    if len(sample_id.split("-")) == 2:
        tissue = sample_id.split("-")[1]
    else:
        tissue = sample_id.split("-")[1]

    exp = sample_id.split("-")[0]

    outfolder = os.path.join(
        cohorts_folder,
        cohort_id,
        exp,
        f'FastQC_{tissue}',
        f'{patient}-{sample_id}_fastqc'
    )

    try:
        os.makedirs(outfolder)
    except OSError:
        print(f'Folder {outfolder} exists')

    data_in = os.path.join(cohorts_folder, cohort_id, 'tmp', sample_id + '_fastqc/fastqc_data.txt')
    data_out = os.path.join(outfolder, 'fastqc_data.txt')
    data_out_gc = os.path.join(outfolder, patient + '-' + sample_id + '.fastqc_filtered.per_sequence_gc_content.txt')

    return data_in, data_out, data_out_gc

def fastqc_gc_content(
        patient: str,
        data_in: str,
        data_out: str,
        data_out_gc: str
):
    """
    Take raw fastqc data, change sample name (add patient id in case of overlap) and extract gc content.

    :param patient: patient id
    :param data_in: raw fastqc data file name
    :param data_out: corrected fastq file name
    :param data_out_gc: gc content only file name
    """
    print('Extracting GC content...')

    if not os.path.exists(data_in):
        print(f'File {data_in} doesn\'t exist')
        return None

    with open(data_in, 'r') as infile, \
            open(data_out, 'w') as outf, \
            open(data_out_gc, 'w') as outf_gc:
        # Renaming sample name in the fastqc_data.txt file (adding sample_id)
        write_gc = False

        for n, line in enumerate(infile):

            if n == 3:
                print('Old filename: {}'.format(line))
                line = line.split()
                new_line = "\t".join((line[0], patient + '_' + line[1]))
                print('New filename: {}'.format(new_line))
                outf.writelines(new_line + '\n')

            else:
                # extracting a block with gc content between two strings
                if '>>Per sequence GC content' in line:
                    write_gc = True

                if write_gc and '>>Per sequence GC content' not in line and ">>END_MODULE" not in line:
                    outf_gc.write(line)

                if write_gc and '>>END_MODULE' in line:
                    write_gc = False

                outf.writelines(line)

def fastqc_cohort(
        cohort_id: str,
        cohorts_folder: str,
        base_dir: str,
        patients = None,
        files_type='raw'
):
    """
    Run fastqc analysis postprocessing of the whole cohort.

    :param cohort_id: name of cohort (e.g. "Overman_PAAD")
    :param cohorts_folder: directory where you want to store unpacked and processed files e.g. "/home/sonya/qc"
    :param patients: list of patient names, by default takes all patients
    :param base_dir: where raw archived fastqc files are stored
    :param files_type: part of filename before extension e.g. "*.fastqc_<files_type>.zip"
    """
    data_folder = os.path.join(base_dir, cohort_id)
    patients = patients or os.listdir(data_folder)
    print(f'Patients {patients}')
    print('***** Processing FastQC reports')

    for patient in tqdm(patients):
        samples_fastqc = [
            os.path.basename(x) for x in
            glob.glob(str(Path(base_dir) / cohort_id / patient / f'*.fastqc_{files_type}.zip'))
        ]

        for fastqc_zip in samples_fastqc:
            infile = Path(base_dir) / cohort_id / patient / fastqc_zip
            wdir = Path(cohorts_folder) / cohort_id / 'tmp'

            try:
                os.makedirs(wdir)
            except OSError:
                pass

            print(f'***** Processing patient {patient}, sample {fastqc_zip}')
            extracted_filename = fastqc_unzip(infile, wdir)

            # Rename folder extracted from zip to expected name
            sample_id = os.path.basename(infile).replace(f'.fastqc_{files_type}.zip', '')
            from_path = os.path.join(wdir, extracted_filename)
            to_path = os.path.join(wdir, sample_id+'_fastqc')
            os.rename(from_path, to_path)

            try:
                data_in, data_out, data_out_gc = create_fastqc_dirs(
                    cohort_id=cohort_id,
                    patient=patient,
                    fastqc_zip=fastqc_zip,
                    cohorts_folder=cohorts_folder,
                    files_type = files_type
                )
                print(data_in, data_out, data_out_gc)
                fastqc_gc_content(patient, data_in, data_out, data_out_gc)
                # print('damn')
            except Exception as e:
                print(e)
                pass

            try:
                pass
                fastqc_clean(wdir)
            except Exception:
                pass
def fastqscreen_patient(
        data_folder: str,
        cohort_folder: str,
        project: str,
        patient: str,
        fastqscreen_tar: str
):
    """
    Preprocess fastqscreen reports.

    :param data_folder: path to directory with data to be analyzed (e.g. /uftp/mvp-data/ngs.socket/patients/WU_MA_BRCA)
    :param cohort_folder: path to directory in local server, were output data will be kept
    :param project: project name
    :param patient: patient name
    :param fastqscreen_tar: tar file with fasqscreen results
    :return: fastq screen reports for patient
    """
    sample_id = fastqscreen_tar.replace(".fastqscreen.tar.gz", "")
    exp = sample_id.split("-")[0]
    tissue = sample_id.split("-")[1]

    with tarfile.open(os.path.join(data_folder, patient, fastqscreen_tar)) as tar:
        members = tar.getmembers()

        for member_info in members:
            member = tar.getmember(member_info.name)
            # example: WES/FastqScreen_normal/MC117-WES-normal-D00195:266:HJL5VBCX2:2.results_fastqscreen/MC117-WES
            # -normal-D00195:266:HJL5VBCX2:2_1_screen.txt
            member.name = "/".join(
                [exp, "FastqScreen_" + tissue, "-".join([patient, exp, tissue, member.name.split("/")[0]]),
                 "-".join([patient, member.name.split("/")[1]])])
            tar.extract(member, path=os.path.join(cohort_folder, project))
            
def fastqscreen_cohort(
        data_folder: str,
        cohort_folder: str,
        project: str,
        patients: str
):
    """
    Process fastqscreen reports.

    :param data_folder: path to directory with data to be analyzed (e.g. /uftp/mvp-data/ngs.socket/patients/WU_MA_BRCA)
    :param cohort_folder: path to directory in local server, were output data will be kept
    :param project: project name
    :param patients: patient list
    :return: folders with reports of fastq screen for cohort
    """
    for patient in tqdm(patients):
        print('Processing ' + patient)
        samples_fastqscreen = [
            os.path.basename(x) for x in glob.glob(
                os.path.join(data_folder, patient, "*.fastqscreen.tar.gz"))
        ]

        print("*****Processing FastqScreen reports")
        if not samples_fastqscreen:
            print('No fastqscreen reports available')
            continue

        for fastqscreen_tar in samples_fastqscreen:
            fastqscreen_patient(data_folder, cohort_folder, project, patient, fastqscreen_tar)
def adjust_width(
        cohort_size: int,
        width = None
) -> float:
    """
    Adjust figure height by cohort size.

    :param cohort_size: size of cohort
    :param width: desired width, by default adjusted to cohort size
    :return: adjusted width
    """
    if width is not None:
        width = width
    elif cohort_size < 40:
        width = 10
    else:
        width = cohort_size / 4

    return width


def adjust_height(
        cohort_size: int,
        height = None,
        width: float = 5.0
):
    """
    Adjust figure height by cohort size.

    :param cohort_size: size of cohort
    :param height: desired height of figure, by default adjusted to cohort size
    :param width: width to use, 5.0 by default
    :return: figure size
    """
    if height is not None:
        fig_size = (width, height)
    elif cohort_size < 40:
        fig_size = (width, 10.0)
    else:
        fig_size = (width, cohort_size / 4)

    return fig_size

def run_multiqc(
        cohorts_folder: str,
        cohort_id: str,
        seq_type: str = 'all',
        interpreter_path = None
):
    """
    Run multiqc on a given cohort.

    :param cohorts_folder: directory where you want to store unpacked and processed files e.g. "/home/sonya/qc"
    :param cohort_id: id of the cohort you want to analyze
    :param seq_type: (all|rna|dna) - available sequencing
    :param interpreter_path: path to python interpreter for MultiQC, by default use current interpreter
    """
    assert seq_type in ('all', 'rna', 'dna'), 'Wrong seq_type argument'
    os.chdir(os.path.join(cohorts_folder, cohort_id))
    print('Changed dir to')
    print(Path().absolute())
    qc_dirs = "./WES ./RNASeq".split()

    if seq_type == "rna":
        qc_exist_dirs = [qc_dir for qc_dir in qc_dirs if (os.path.exists(qc_dir)) and ("RNA" in qc_dir)]
    elif seq_type == "dna":
        qc_exist_dirs = [qc_dir for qc_dir in qc_dirs if (os.path.exists(qc_dir)) and ("WES" in qc_dir)]
    else:  # all
        qc_exist_dirs = [qc_dir for qc_dir in qc_dirs if os.path.exists(qc_dir)]

    if len(qc_exist_dirs) == 0:
        raise NameError("No directories containing QC files found")

    if interpreter_path is None:
        interpreter_path = sys.executable

    command = f'{interpreter_path} -m multiqc' \
              f' --interactive -f ' + ' '.join(qc_exist_dirs) + \
              ' -n ' + cohort_id + '-' + seq_type
    print(command)
    subprocess.check_output(command, shell=True)


def intersect_genes_with_pipeline(df, return_genes=False, verbose=True, pipeline_genes=None):
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
    if pipeline_genes==None:
        pipeline_genes = pd.read_csv(
            Path(__file__)
            .resolve()
            .parent.joinpath('databases', 'pipeline_genes.tsv'),
        sep='\t')['Gene'].to_list()
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


def check_log_scaling(df, return_non_scaled=False, return_scaling=False, verbose=True):
    """
    Checks if the provided dataframe is possibly log-scaled based on the mean values.
    
    The function calculates the mean of each column in the dataframe and determines if it 
    might be log-scaled by comparing the maximum of these means to a threshold of 20. If the 
    maximum mean is below 20, the function infers that the data is possibly log-scaled. If 
    return_non_scaled is set to True, it then returns the dataframe with the data reverted 
    back to its original scaling using the inverse of the log2 operation.

    :param df: DataFrame to check for log-scaling.
    :param return_non_scaled: A flag to decide whether to return the dataframe with the 
                              data reverted back to its original scaling, if it's determined 
                              to be log-scaled. Default is False.
    :return: If return_non_scaled is True and the data is determined to be log-scaled, 
             returns the dataframe with the data reverted back to its original scale. 
             Otherwise, None.
    
    Example:
    --------
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [6, 7, 8, 9, 10],
    ...     'C': [11, 12, 13, 14, 15]
    ... })
    >>> check_log_scaling(np.log2(data+1), True)
    Maximum of average gene values is below 20.
    Probably Log-scaled
          A    B    C
    0  1.0  6.0  11.0
    1  2.0  7.0  12.0
    2  3.0  8.0  13.0
    3  4.0  9.0  14.0
    4  5.0 10.0  15.0
    """

    if df.mean().max() > 20:
        if verbose:
            print('OK')
        if return_scaling:
            return True
    else:
        if verbose:
            print('Maximum of average gene values is below 20.\nProbably Log-scaled')
        if return_non_scaled:
            return np.exp2(df) - 1
        if return_scaling:
            return False

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