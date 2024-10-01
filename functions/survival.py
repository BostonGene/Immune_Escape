import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

black_color = 'black'

from functions.plotting import (lin_colors, axis_net)
from functions.utils import (round_to_1,
    to_common_samples,
    get_pvalue_string)


class SurvivalData:
    def __init__(
        self,
        durations: pd.Series,
        events: pd.Series,
        units='Days',
        survival_type='OS',
        max_time=None,
    ):
        """
        A basic class for survival data storage
        :param durations: Series with survival in units
        :param events: pd.Series with events (True: event, False: censored)
        :param units: Units used in duration. Typically: Days/Weeks/Months/Years.
        :param survival_type: Str to display
        :param max_time: Max time to cut the durations column
        """
        self.duration, self.events = to_common_samples(
            [durations.dropna(), events.dropna()]
        )
        if len(self.duration) == 0:
            raise Exception('Different id\'s provided')

        self.duration = self.duration.astype(float)
        try:
            self.events = self.events.astype(float).astype(bool)
        except Exception as e:
            raise Exception('Bad censorship ' + str(e))

        if max_time:
            self.events = np.logical_and(self.duration < max_time, self.events)
            self.duration = self.duration.clip(upper=max_time)

        self.units = units
        self.survival_type = survival_type
        self.index = self.duration.index

        self.data = pd.DataFrame({'duration': self.duration, 'events': self.events})

    def plot(self, ax=None, label='Samples', **kwargs):
        """
        Simple KM plot of the all samples
        :param ax:
        :param label:
        :param kwargs:
        :return:
        """
        kwargs['palette'] = kwargs.get('palette', {label: black_color})
        kaplan_meier_plot(
            pd.Series(label, index=self.index), self, ax=ax, pvalue=False, **kwargs
        )

    def intersect_index(self, data):
        """
        Returns a subset of data with common indexes
        :param data: Series/DataFrame
        :return:
        """
        data_c = data.reindex(self.index).dropna(how='all')
        discarded_samples_am = len(data) - len(data_c)
        if discarded_samples_am:
            warnings.warn(
                f'{discarded_samples_am} out of {len(data)} discarded due to no survival annotation'
            )
        return data_c

    def event_at_time(self, time):
        """
        Returns a series with Event/Censored/No event for each sample at a specific time
        :param time:
        :return:
        """
        return pd.concat(
            [
                self.data[self.data.duration <= time].events.map(
                    {True: 'Event', False: 'Censored'}
                ),
                pd.Series('No event', index=self.data[self.data.duration > time].index),
            ]
        )


def prepare_survival_annotation(
    durations: pd.Series,
    events: pd.Series,
    max_time=None,
    in_units='Days',
    out_units='Months',
    survival_type='OS',
) -> SurvivalData:
    """
    Prepare SurvivalData with survival annotation suitable for kaplan_meier_plot()
    :param durations: Series with survival in days
    :param events: Series with Death events as True/False or 1/0
    :param max_time: Limit survival time and events (in out units)
    :param in_units: Should be Days, others not implemented
    :param out_units: To convert days into Weeks/Months/Years
    :param survival_type:
    :return: pd.DataFrame, index - patients, columns - ['duration', 'events']
    """
    durations_c = durations.copy()

    if in_units == 'Days':
        if out_units == 'Days':
            pass
        elif out_units == 'Weeks':
            durations_c /= 7.0
        elif out_units == 'Months':
            durations_c = durations_c / 365.25 * 12
        elif out_units == 'Years':
            durations_c /= 365.25
        else:
            raise NotImplementedError(
                f'{in_units} not supported. Try constructing SurvivalData by yourself'
            )
    else:
        raise Exception('Not implemented')

    return SurvivalData(durations_c, events, out_units, survival_type, max_time)


def kaplan_meier_plot(
    groups: pd.Series,
    survival: SurvivalData,
    loglogs=False,
    title='',
    palette=None,
    pvalue=True,
    ax=None,
    figsize=(4, 4.5),
    p_digits=25,
    order=None,
    cmap=plt.cm.rainbow,
    max_time=None,
    legend='in',
    ci_show=False,
    title_n_samples=True,
    add_at_risk=False,
    weightings=None,
    **kwargs,
):
    """
    Plot survival curves for each group in "groups"
    :param groups: pd.Series, indexed with sample names, groups as values
    :param survival: SurvivalData
    :param loglogs: bool, a quick graphical check of the Hazard Ratio assumption - your curves must not intersect
    :param title: str, plot title
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param pvalue: bool, whether to perform a logrank_test on groups. Adds p-value to title if True
    :param ax: matplotlib axis, axis to plot on
    :param figsize: (float, float), figure size in inches
    :param p_digits: int, number of digits to round p value to
    :param order: plotting order of survival curves of groups
    :param cmap: matplotlib colormap, if palette is not given, it will be generated based on this colomap
    :param max_time if specified - will be used as max time
    :param legend where legend is to be plotted
    :param add_at_risk: add patients at risk in the bottom. It is accounted in the plot size!
    :param weightings: weights used in logrank-test (see logrank_test function)
    :param title_n_samples: add N samples to title
    :param ci_show: show confidence intervals
    :return: matplotlib axis
    """

    from matplotlib.ticker import FuncFormatter
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    kmf = KaplanMeierFitter()

    emt = False
    if max_time is None:
        max_time = 0
        emt = True

    groups_c = survival.intersect_index(groups)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if order is None:
        order = list(sorted(groups_c.dropna().unique()))

    groups_c = groups_c[groups_c.isin(order)]

    if palette is None:
        c_palette = lin_colors(pd.Series(order), cmap=cmap)
    else:
        c_palette = copy.copy(palette)

    kmfs = []
    for group_name in order:
        samps = groups_c[groups_c == group_name]
        if len(samps):
            kmf.fit(
                survival.duration[samps.index], survival.events[samps.index], label=''
            )
            if loglogs:
                kmf.plot_loglogs(
                    ax=ax,
                    show_censors=True,
                    c=c_palette[group_name],
                    label=str(group_name),
                )
            else:
                kmf.plot_survival_function(
                    ax=ax,
                    ci_show=ci_show,
                    show_censors=True,
                    c=c_palette[group_name],
                    label=str(group_name),
                )

            if emt:
                max_time = max(max_time, survival.duration[samps.index].max())

            kmf._label = group_name
            kmfs.append(copy.copy(kmf))

    if len(title):
        title += ', '
    if title_n_samples:
        title += f'N={len(groups_c)}'

    if not loglogs:
        if pvalue:
            if len(title):
                title += '\n'
            lgt_pvalue = logrank_test(groups_c, survival, weightings)
            title += get_pvalue_string(lgt_pvalue, p_digits)

        if hasattr(groups, 'name') and groups.name is not None:
            ax.set_ylabel(f'{survival.survival_type}-{groups.name}, %')
        else:
            ax.set_ylabel(f'{survival.survival_type}, %')

        ax.set_xlim(max_time * -0.05, max_time * 1.05)
        ax.set_ylim((0, 1.05))

        ax.set_xlabel(kwargs.get('xlabel', survival.units))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))

        if add_at_risk:
            add_at_risk_counts(*kmfs, ax=ax)

    if legend == 'out':
        ax.legend(scatterpoints=1, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
    elif legend == 'in':
        ax.legend(scatterpoints=1, loc='best')
    else:
        ax.legend_.remove()
    ax.set_title(title)

    return ax

def kaplan_meier_stratified(
    groups: pd.Series,
    survival: SurvivalData,
    strat: pd.Series,
    strat_order=None,
    title='',
    h_plots=None,
    plot_size=(4, 4),
    **kwargs,
):
    """
    For each unique value in "strat" plot survival curves of each group (split into subplots by values in strat)
    :param groups: pd.Series, indexed with sample names, groups as values
    :param survival: SurvivalData
    :param strat: pd.Series, indexed with sample names, strata as values
    :param strat_order: groups order plotting
    :param title: str, plot title
    :param h_plots: int, number of subplots in a row
    :param plot_size: float, figure side size in inches
    :param kwargs:
    :return: matplotlib axis
    """
    c_groups, c_strat = to_common_samples([groups, strat])
    if strat_order is None:
        strat_order = np.sort(c_strat.dropna().unique())
    c_groups, c_strat = to_common_samples(
        [c_groups, c_strat[c_strat.isin(strat_order)]]
    )

    if h_plots is None:
        h_plots = int(np.ceil(np.sqrt(len(strat_order))))
    w_plots = (len(strat_order) + h_plots - 1) // h_plots

    af = axis_net(
        h_plots,
        w_plots,
        x_len=plot_size[0],
        y_len=plot_size[1],
        title=title,
        title_y=1.02,
    )
    for st in strat_order:
        ax = next(af)
        kaplan_meier_plot(
            c_groups[c_strat[c_strat == st].index],
            survival,
            ax=ax,
            title=str(st),
            **kwargs,
        )
    plt.tight_layout()
    return af


def fit_kmf(survival: SurvivalData):
    """
    Fits a kmf model
    :param survival: KaplanMeierFitter
    :return:
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(survival.duration, event_observed=survival.events)
    return kmf


def logrank_test(groups: pd.Series, survival: SurvivalData, weightings=None, alpha=0.95):
    """
    Perform multivariate logrank test. See lifelines.multivariate_logrank_test documentation for more
    :param groups: pd.Series, indexed with sample names, groups as values
    :param survival: SurvivalData
    :param weightings: apply a weighted logrank test: options are “wilcoxon” for Wilcoxon (also known as Breslow),
    “tarone-ware” for Tarone-Ware, “peto” for Peto test and “fleming-harrington” for Fleming-Harrington test. These are
    useful for testing for early or late differences in the survival curve.  (see https://lifelines.readthedocs.io/en/la
    test/lifelines.statistics.html)
    :param alpha:
    :return: float, p value
    """
    from lifelines.statistics import multivariate_logrank_test

    groups_c = survival.intersect_index(groups)

    if len(groups_c.unique()) < 2:
        warnings.warn('Less than 2 groups provided')
        return 1

    return multivariate_logrank_test(
        event_durations=survival.duration[groups_c.index],
        groups=groups_c,
        event_observed=survival.events[groups_c.index],
        alpha=alpha,
        weightings=weightings,
    ).p_value


def simple_multivar_cox_reg(
    survival: SurvivalData,
    numerical_list=None,
    categorical_list=None,
    return_data=False,
    verbose=True,
):
    """
    Performs simple CPH regression with categorical variables auto dummying and returns a model
    :param survival: SurvivalData
    :param numerical_list: a list with DataFrames/Series with numerical data
    :param categorical_list: a list with DataFrames/Series with categorical data
    :param return_data: boolearn, to return prepared datasets for lifelines functions
    :param verbose: boolean, to write a ton of text
    :return:
    """
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()

    if numerical_list is None or not len(numerical_list):
        numerical_list = []
    else:
        survival_f, *numerical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in numerical_list]
        )

    if categorical_list is None or not len(categorical_list):
        categorical_list = []
    else:
        survival_f, *categorical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in categorical_list]
        )

    dat = pd.concat(
        list(numerical_list)
        + [
            pd.get_dummies(x.dropna(), drop_first=True, prefix=x.name)
            for x in categorical_list
        ]
        + [survival_f],
        axis=1,
    ).dropna()

    cph.fit(dat, duration_col='duration', event_col='events', show_progress=verbose)

    if verbose:
        cph.print_summary()

    if return_data:
        return cph, dat

    return cph


def simple_multivar_cox_reg_check_hazard_ratio_assumption(
    survival: SurvivalData,
    numerical_list=None,
    categorical_list=None,
    verbose=True,
):
    """
    Function to perform the check of proportional hazard ratio assumption (for more information see: https://lifelines.r
    eadthedocs.io/en/latest/jupyter_notebooks/Proportional%20hazard%20assumption.html#)
    :param survival: SurvivalData
    :param numerical_list: a list with DataFrames/Series with numerical data
    :param categorical_list: a list with DataFrames/Series with categorical data
    :param verbose:
    :return:
    """
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()

    if numerical_list is None or not len(numerical_list):
        numerical_list = []
    else:
        survival_f, *numerical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in numerical_list]
        )

    if categorical_list is None or not len(categorical_list):
        categorical_list = []
    else:
        survival_f, *categorical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in categorical_list]
        )

    dat = pd.concat(
        list(numerical_list)
        + [
            pd.get_dummies(x.dropna(), drop_first=True, prefix=x.name)
            for x in categorical_list
        ]
        + [survival_f],
        axis=1,
    ).dropna()

    cph.fit(dat, duration_col='duration', event_col='events', show_progress=verbose)
    cph.check_assumptions(dat)


def calculate_vif(df):
    """
    Calculates the Variance Inflation Factor (VIF) for each numeric variable in a DataFrame.
    
    :param df: DataFrame
        A DataFrame that should contain only numeric variables. If there are any categorical variables, they should be preprocessed beforehand, for example, using pd.get_dummies() method.
        
    VIF is a measure of multicollinearity in a set of multiple regression variables. A high VIF indicates a high correlation of a variable with other variables in the set, which can impair the precision of estimated coefficients in regression models.
    
    :return: DataFrame
        A DataFrame containing two columns: 'feature' and 'VIF'. The 'feature' column lists the names of the variables, and the 'VIF' column contains the corresponding Variance Inflation Factor values.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = {"feature": df.columns}
    # Calculating VIF for each variable
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

    return pd.DataFrame(vif_data)

def scale_series(series,feature_range=(0, 1)):
    """
    Scales a pandas Series to a specified range using Min-Max scaling.

    This function takes a pandas Series and scales it so that the values fall within a specified range, typically between 0 and 1. It uses the Min-Max scaling approach from scikit-learn, which transforms each value in the Series based on the minimum and maximum values in the Series.

    Parameters:
    series : pandas.Series
        The Series to be scaled. It should be a numerical Series.
    
    feature_range : tuple, optional
        The desired range of transformed data. Default is (0, 1), meaning that after scaling, the minimum value in the Series will be 0 and the maximum will be 1.
    
    Returns:
    pandas.Series
        The scaled Series. It retains the original index and name of the Series.
    """
    from sklearn.preprocessing import MinMaxScaler
    name = series.name
    scaler = MinMaxScaler(feature_range=feature_range)
    series_2d = series.values.reshape(-1, 1)
    scaled_series_2d = scaler.fit_transform(series_2d)
    scaled_series = pd.Series(scaled_series_2d.flatten(), index=series.index)
    scaled_series.name =name
    return scaled_series

def find_worst_groups(reference_list,categorical_list, survival_f, warnigns, dat):
    """
    Finds the group within a categorical variable that has the worst survival.

    :param reference_list: list of references
        A references for variables
    :param categorical_list: list of pandas.Series
        List of categorical variables
    :param survival_f: pandas.DataFrame
        The DataFrame containing survival data, with columns for duration and event status.
    :param warnings: boolean
        Switcher to write warnings or not
    :param dat: pandas DataFrame
        Data for cox regression

    :return: list of str
        List of the names of the groups within the categorical variables with the worst survival.
    """
    from lifelines import CoxPHFitter
    if reference_list and None not in reference_list and len(reference_list)==len(categorical_list):
        ref_list = []
        for i in range(len(categorical_list)):
            ser = categorical_list[i]
            g = reference_list[i]
            if g in ser.unique():
                ref_list.append(g)
            else:
                if warnigns:
                    print(f'Group name {g} is not in categories for {ser.name}. Group with the worst survival is used instead')
                ref_list.append(None)
    else: 
        if reference_list and len(reference_list)!=len(categorical_list) and warnings:
            print('Reference list length is not equal to categorical list length. Reference list is remade with groups with worst survival as references.')
        ref_list = [None]*len(categorical_list) if not reference_list or len(reference_list)!=len(categorical_list) else [i for i in reference_list]
        for j,x in enumerate(categorical_list):
            group_name = ref_list[j]
            if not group_name or not group_name in x.unique():
                if group_name:
                    print(f'Group name {group_name} is not in categories for {x.name}. Group with the worst survival is used instead')
                cols = [i for i in dat.columns if x.name in i][1:]+survival_f.columns.to_list()
                cox = CoxPHFitter()
                cox = cox.fit(dat[cols], duration_col='duration', event_col='events', show_progress=False).summary.coef.sort_values(ascending=False)
                coef0, group_name0 = cox[0], cox.index[0]
                cols = [i for i in dat.columns if x.name in i][:-1]+survival_f.columns.to_list()
                cox = CoxPHFitter()
                cox = cox.fit(dat[cols], duration_col='duration', event_col='events', show_progress=False).summary.coef.sort_values(ascending=False)
                coef1, group_name1 = cox[0], cox.index[0]
                group_name = group_name0 if coef0>=coef1 else group_name1
                ref_list[j] = group_name[len(x.name)+1:]
    return [f"{categorical_list[i].name}_{group_name}" for i,group_name in enumerate(ref_list)]  

def create_dummy_formula(formula, dat,warnings, categorical_list, numerical_list):
    """
    Creates a processable formula for dummies

    :param formula: str
        Formula that the user wrote
    :param dat: pandas DataFrame
        Data for cox regression
    :param warnings: boolean
        Switcher to write warnings or not
    :param categorical_list: list of pandas series
        List to know what categories are present
    :param numerical_list: list of pandas series
        List to know what num vars are present
        
    :return: str
        Updated formula with dummy variables
    """
    import re
    from itertools import combinations
    import traceback
    try: 
            formula = formula.replace(' +', '+').replace(' *', '*').replace('+ ', '+').replace('* ', '*').split('+')
            formula_no_inter = []
            formula_w_inter = []
            for variable in formula:
                if "*" not in variable:
                    to_add = [i for i in dat.columns[:-2] if re.match(f'^{variable}_', i) or variable==i]
                    if warnings and len(to_add)==0:
                        print(f'Variable {variable} not in the lists.')
                    formula_no_inter+=to_add
                else:
                    for var in variable.split('*'):
                        to_add = [i for i in dat.columns[:-2] if re.match(f'^{var}_', i) or var==i]
                        if warnings and len(to_add)==0:
                            print(f'Variable {var} not in the lists.')
                        formula_w_inter+=to_add
            formula_no_inter = [i for i in formula_no_inter if i not in formula_w_inter]
            missing_cols = []
            for col in dat.columns[:-2]:
                if col not in formula_no_inter+formula_w_inter:
                    formula_no_inter.append(col)
                    if warnings:
                        missing_cols.append(col)    
            if warnings:
                print(f'\n The following columns were missing in the formula and added with no interactions:')
                print(', '.join(missing_cols)+'\n')
            final_formula = ' + '.join(formula_no_inter)+' + '
            formula_w_inter_dict = {x.name:[i for i in formula_w_inter if x.name in i] for x in categorical_list+numerical_list if len([i for i in formula_w_inter if x.name in i])>0}
            for var1, var2 in combinations(list(formula_w_inter_dict.keys()), 2):
                for group in formula_w_inter_dict[var1]:
                    final_formula+=f'{group} * '+' * '.join(formula_w_inter_dict[var2])+' + '
            final_formula = final_formula.replace('-','_').strip(' + ')
            dat.columns = dat.columns.map(lambda x: x.replace('-','_'))
            if warnings:
                print('\nFinal formula is as follows:')
                print(final_formula+'\n')
    except Exception as e:
            if warnings:
                print('\nFailed to process the formula, will not use it. Here\'s error info:')
                print(traceback.format_exc())
            final_formula=None
    return final_formula
from lifelines.statistics import proportional_hazard_test


def simple_multivar_cox_reg_with_references(
    survival: SurvivalData,
    numerical_list=None,
    categorical_list=None,
    return_data=False,
    verbose=True,
    warnings=True,
    reference_list = None,
    show_schoenfeld_plots = False,
    schoenfeld_pvalue = 0.01,
    scale_numerical = True,
    formula = None,
    rewrite_formula=True,
    strata = None,
    return_warnings = False
):
    """
    Performs simple Cox Proportional Hazards (CPH) regression with automated dummy variable creation for categorical variables and returns a fitted model.

    :param survival: SurvivalData
        The survival data to be used in the model.
    :param numerical_list: list
        A list of DataFrames/Series containing numerical data.
    :param categorical_list: list
        A list of DataFrames/Series containing categorical data.
    :param return_data: bool
        If True, returns the datasets prepared for lifelines functions.
    :param verbose: bool
        If True, writes a detailed CPH model summary.
    :param warnings: bool
        If True, writes warnings about possible assumption violations and references. It is recommended to keep this switched on for model diagnostics.
    :param reference_list: list
        A list of group names to be used as references for the categorical data, in the same order as the categories in 'categorical_list'. Use "None" for a category to use the group with the worst survival as a reference. If None, the group with the worst survival is used for all categories.
    :param show_schoenfeld_plots: bool
        If True, shows Schoenfeld residual scatter plots.
    :param schoenfeld_pvalue: float
        P-value threshold for the Schoenfeld residual test. Default is 0.01.
    :param scale_numerical: bool
        If True, scales numerical variables from 0 to 1 for better CPH performance. This scaling is recommended for better interpretability of model coefficients and comparison between variables. Note: this function does not exclude outliers or perform other types of scaling (MAD-, Z-, log, etc.); the variable distribution remains unchanged.
    :param formula: str
        A string specifying the formula for the Cox model. Use variable names (from 'numerical_list' and/or 'categorical_list') with "+" (no interaction) or "*" (indicating interaction between variables). This function automatically transforms categories into dummy variables. By default, variable names in the formula are applied to dummies.
        - For a variable with 3 or more categories where interaction is desired (e.g., Stages I-IV), specify a dummy name instead of a variable name. A dummy name is composed of the pandas series name followed by an underscore and the category name.
        - Note: Avoid overloading the model with many interactions and use distinctive variable names to prevent errors.
        :param rewrite_formula: bool
        A flag to switch off if one wants to write down formula for dummies and num values manually.
    :param strata: str or a list of str
        A string for category or a list of ones to stratify Cox regression. Handy when assumptions are not satisfied for the variable. Similar to strat expression in R http://courses.washington.edu/b515/l17.pdf
    :return: tuple
        Returns a CoxPHFitter model and, optionally, the dataset if 'return_data' is True.
    """
    from lifelines import CoxPHFitter
    import traceback
    from math import comb
    return_warnings_dict = {}
    cph = CoxPHFitter()

    if numerical_list is None or not len(numerical_list):
        numerical_list = []
    else:
        survival_f, *numerical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in numerical_list]
        )

    if categorical_list is None or not len(categorical_list):
        categorical_list = []
    else:
        survival_f, *categorical_list = to_common_samples(
            [survival.data] + [x.dropna() for x in categorical_list]
        )
    numerical_list = [scale_series(i) for i in list(numerical_list)] if scale_numerical and len(numerical_list)>0 else list(numerical_list)  
    dat = pd.concat(
        numerical_list
        + [
            pd.get_dummies(x.dropna(), prefix=x.name)
            for x in categorical_list
        ]
        + [survival_f],
        axis=1,
    ).dropna()
    ref_list = find_worst_groups(reference_list,categorical_list, survival_f, warnings, dat)
    ref_list = new_list = [ref_list[i] if 'None' not in ref_list[i] else f"{[j for j in dat.columns if categorical_list[i].name in j][0]}" for i in range(len(ref_list))]
    dat = dat.drop(columns = ref_list)       
    
    if len(dat.columns)>3:
        vif_df = calculate_vif(dat[dat.columns[:-2]].astype('float'))
        vif_high = vif_df[vif_df.VIF>=10]
        if len(vif_high)>0 and warnings:
            print('\nThe following features have high VIF:')
            print('\n'.join([f"{vif_high.loc[i].feature}, with VIF {round(vif_high.loc[i].VIF,2)}" for i in vif_high.index])+'\n')
            return_warnings_dict['vif_fail'] = [vif_high.loc[i].feature for i in vif_high.index]
        elif len(vif_high)==0 and warnings:
            print('\nAll features have VIF less than 10\n')
    else:
        if warnings:
            print('\nUnable to estimate VIF for univariate regression\n')

    final_formula = create_dummy_formula(formula, dat,warnings, categorical_list, numerical_list) if formula and rewrite_formula else formula
    if formula:
        added_interactions = sum([len(i.split(' * ')) for i in final_formula.split(' + ') if ' * ' in i])
        added_interactions = sum(comb(added_interactions, k) for k in range(1, added_interactions+1))
    else:
        added_interactions = 0
    ev = len(dat.events[dat.events==True])
    if warnings:
        print(f'\nNumber of events in the data is {ev} out of {len(dat.events)}, or {ev/len(dat.events)*100}%')
    if ev/(len(dat.columns[:-2])+added_interactions)<1:
        final_formula = None
        if warnings:
            print(f'\nFormula was too long to fit the model. Number of interaction added is {added_interactions}. Discarding it\n')
        added_interactions = 0
    elif ev/(len(dat.columns[:-2])+added_interactions)>=10 and warnings:
        if ev/len(dat.columns[:-2]) >= 10:
            print('Event number fits into one in ten rule. \n')
        else:
            print('Event number is less than 10. There is a risk of overfitting and unreliable estimations of variables (see one in ten rule). Consider decreasing variable number\n')
    cph.fit(dat, duration_col='duration', event_col='events', show_progress=verbose, formula = final_formula, strata=strata)
    if warnings:
        cph.check_assumptions(dat,show_plots=show_schoenfeld_plots, p_value_threshold=schoenfeld_pvalue, advice=False, plot_n_bootstraps=3)
        results_rank = proportional_hazard_test(cph, dat, time_transform='rank')
        results_rank = list(results_rank.summary['p'][results_rank.summary['p']<schoenfeld_pvalue].index)
        results_km = proportional_hazard_test(cph, dat, time_transform='km')
        results_km = list(results_km.summary['p'][results_km.summary['p']<schoenfeld_pvalue].index)
        results = set(results_km+results_rank)
        return_warnings_dict['sch_fail'] = results
    if verbose:
        cph.print_summary()
    if return_warnings:
        return cph, dat, return_warnings_dict
    if return_data:
        return cph, dat
    return cph
    

def plot_cph_model(cph, col_rename_dict=None, good_color='limegreen', bad_color='salmon', title=None, survival_type=None):
    """
    Plots the coefficients from a Cox Proportional Hazards (CPH) model along with their significance. This function visualizes the effect size of each variable in the model, highlighting significant variables with different colors based on their impact on the hazard (positive or negative).

    Parameters:
    cph : CoxPHFitter object
        A fitted CoxPHFitter object from the lifelines library.
    
    col_rename_dict : dict, optional
        A dictionary to rename the column labels in the plot. The keys should be the original column names, and the values should be the new names. If None, original column names are used.
    
    good_color : str, optional
        Color used to highlight significant coefficients that have a negative effect on the hazard (indicating a better prognosis). Default is 'limegreen'.
    
    bad_color : str, optional
        Color used to highlight significant coefficients that have a positive effect on the hazard (indicating a worse prognosis). Default is 'salmon'.
    
    title : str, optional
        Title for the plot. If None, a default title including the model's Harrell's C-index is used.
    
    survival_type : str, optional
        Additional text to append to the plot title, typically used to specify the type of survival analysis.

    Returns:
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    
    ax : matplotlib.axes.Axes
        The axes object of the plot.
    """
    from matplotlib.lines import Line2D
    cols = cph.summary.index.intersection(col_rename_dict.keys()) if col_rename_dict else cph.summary.index
    fig, ax = plt.subplots(figsize=(6, (0.4)*len(cols)))
    cph.plot(columns=cols, ax=ax, )

    # Set y-tick labels according to provided dictionary
    if col_rename_dict:
        ax.set_yticklabels(labels=[col_rename_dict.get(col, col) for col in cols[::-1]])
    cols = cols[::-1]
    # Logic for custom lines based on p-values and coefficients
    for l in range(len(ax.lines)):
        line = ax.lines[l].get_data()
        for i in range(len(cols)):
            col = cols[i]
            if cph.summary.loc[col, 'p'] <= 0.05:
                color = bad_color if cph.summary.loc[col, 'coef'] > 0 else good_color
                color = 'black' if cph.summary.loc[col, 'coef'] == 0 else color
                xs = [line[0][i], ax.lines[1].get_data()[0][i]]
                ys = [line[1][i], ax.lines[1].get_data()[1][i]]
                plt.plot(xs, ys, c=color, linewidth=2.5)
                
    # Custom legend for significance
    custom_lines = [
        Line2D([0], [0], color=good_color, marker='s', markersize=8, linestyle='None', linewidth=2.5),
        Line2D([0], [0], color='black', marker='s', markersize=8, linestyle='None', linewidth=2.5),
        Line2D([0], [0], color=bad_color, marker='s', markersize=8, linestyle='None', linewidth=2.5)
    ]

    ax.legend(custom_lines, ['significant, better prognosis', 'non-significant', 'significant, worse prognosis'],bbox_to_anchor=(1,1))

    # Final adjustments before returning the plot
    ax.xaxis.set_ticks_position('bottom')
    # Annotation for risk direction

    
    title = f'{title},\nHarrell\'s C-index = {round(cph.concordance_index_,2)}' if title else f'Harrell\'s C-index = {round(cph.concordance_index_,2)}'
    title+=",\n"+survival_type if survival_type else title
    ax.set_title(title) 
    return fig, ax