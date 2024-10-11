import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functions.utils import item_series, to_common_samples, sort_by_terms_order
import matplotlib.patches as mpatches
from adjustText import adjust_text
from scipy import stats 
from functions.utils import get_pvalue_string
from matplotlib.lines import Line2D
from functions.utils import round_to_1
from matplotlib.colors import LinearSegmentedColormap, to_hex, rgb_to_hsv
import distinctipy

default_cmap = LinearSegmentedColormap.from_list("default_cmap", ["navy", "white", "crimson"])

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

def palette_from_series(series, exclude_colors=['black', 'white'], pastel_factor=0, n_attempts=1000, colorblind_type='Deuteranomaly', rng=42, color_format='HEX'):
    '''
    Generates a color palette from a pandas series, ensuring distinct colors for each category.

    Parameters:
    series (pandas.Series): A series with categories.
    exclude_colors (list): List of colors to exclude.
    pastel_factor (float): If >0, generates paler colors.
    n_attempts (int): Number of attempts to generate distinct colors.
    colorblind_type (str): Type of colorblindness to account for.
    rng (int): Random number generator seed.
    color_format (str): The format of the color output ('HEX', 'RGBA', 'HSL').

    Returns:
    dict: A dictionary mapping categories to colors.
    '''

    # Convert excluded colors from HEX to RGB
    exclude_colors_rgb = [tuple(int(to_hex(hex_color).strip('#')[i:i+2], 16)/255 for i in (0, 2, 4)) for hex_color in exclude_colors] if len(exclude_colors)!=0 else exclude_colors

    # Get unique categories
    categories = series.unique()
    n_colors = len(categories)

    # Get distinct colors
    colors = distinctipy.get_colors(n_colors=n_colors, exclude_colors=exclude_colors_rgb, pastel_factor=pastel_factor, n_attempts=n_attempts, colorblind_type=colorblind_type, rng=rng)

    # Format colors according to the specified format
    if color_format == 'HEX':
        color_dict = {category: to_hex(color) for category, color in zip(categories, colors)}
    elif color_format == 'RGBA':
        color_dict = {category: color for category, color in zip(categories, colors)}
    elif color_format == 'HSL':
        color_dict = {category: rgb_to_hsv(*color) for category, color in zip(categories, colors)}
    else:
        raise ValueError("Unsupported color format. Choose 'HEX', 'RGBA', or 'HSL'.")

    return color_dict
    
def simple_scatter(x, y, ax=None, title='', color='b', figsize=(5, 5), s=20, **kwargs):
    """
    Plot a scatter for 2 vectors. Only samples with common indexes are plotted.
    If color is a pd.Series - it will be used to color the dots
    :param x: pd.Series, numerical values
    :param y: pd.Series, numerical values
    :param ax: matplotlib axis, axis to plot on
    :param title: str, plot title
    :param color: str, color to use for points
    :param figsize: (float, float), figure size in inches
    :param s: float, size of points
    :param alpha: float, alpha of points
    :param marker: str, marker to use for points
    :param linewidth: float, width of marker borders
    :param edgecolor: str, color of marker borders
    :return: matplotlib axis
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    try:
        c_x, c_y, c_color = to_common_samples([x, y, color])
    except Exception:
        c_x, c_y = to_common_samples([x, y])
        c_color = color

    ax.set_title(title)

    ax.scatter(c_x, c_y, color=c_color, s=s, **kwargs)

    if hasattr(x, 'name'):
        ax.set_xlabel(x.name)
    if hasattr(y, 'name'):
        ax.set_ylabel(y.name)

    return ax


def simple_palette_scatter(
    x: pd.Series,
    y: pd.Series,
    grouping: pd.Series,
    palette: Optional[Dict[Any, str]] = None,
    order: List[Any] = None,
    centroids: bool = False,
    confidence_level: Union[float, bool] = False,
    legend: bool = 'out',
    patch_size: int = 10,
    centroid_complement_color: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Plot a scatter for 2 vectors, coloring by grouping.
    Only samples with common indexes are plotted.

    See Also
    --------------------
    plotting.simple_scatter

    Parameters
    --------------------
    x: pd.Series
        numerical values
    y: pd.Series
        numerical values
    grouping: pd.Series
        which group each sample belongs to
    palette: dict
        palette for plotting. Keys are unique values from groups, entries are color hexes
    order: list
        order to plot the entries in. Contains ordered unique values from grouping
    centroids: bool
        whether to plot centroids of each group
    confidence_level: (0, 1) or False, default: False
        Confidence level, based on standard deviation, must be from 0 to 1
    legend: bool
        whether to plot legend
    patch_size: float
        size of legend
    centroid_complement_color: bool
        whether to plot centroids in complement color
    ax: plt
        axes to plot on

    Returns
    --------------------
    matplotlib axis
    """

    if palette is None:
        palette = lin_colors(grouping)

    if order is None:
        order = np.sort(list(palette.keys()))

    c_grouping, c_x, c_y = to_common_samples(
        [grouping[sort_by_terms_order(grouping, order)], x, y]
    )

    patch_location = 2
    if 'loc' in kwargs:
        patch_location = kwargs.pop('loc')

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(kwargs.get('figsize', (4, 4))))

    kwargs['marker'] = kwargs.get('marker', 'o')
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'black')
    kwargs['linewidth'] = kwargs.get('linewidth', 0)

    for label in order:
        samps = c_grouping[c_grouping == label].index
        simple_scatter(c_x[samps], c_y[samps], color=palette[label], ax=ax, **kwargs)
        handles = [mpatches.Patch(color=palette[label], label=label) for label in order]

    if centroids:
        c_color = 'black'
        for label in order:
            samps = c_grouping[c_grouping == label].index
            mean_x = c_x[samps].mean()
            mean_y = c_y[samps].mean()
            if centroid_complement_color:
                c_color = complementary_color(palette[label])

            if 's' in kwargs:
                s = kwargs['s']
            else:
                s = 20
            ax.scatter(
                mean_x,
                mean_y,
                marker='*',
                lw=1.5,
                s=s * 10,
                edgecolor=c_color,
                color=palette[label],
            )

    if 0 < confidence_level < 1:
        import scipy.stats as st

        sigma = st.norm.ppf(
            1 - (1 - confidence_level) / 2
        )  # confidence_level from 0 to 1
        for label in order:
            samps = c_grouping[c_grouping == label].index
            confidence_ellipse(
                x=c_x[samps],
                y=c_y[samps],
                ax=ax,
                n_std=sigma,
                edgecolor=palette[label],
                ls='--',
            )
            handles = [
                mpatches.Patch(
                    color=palette[label],
                    label=f'{label} ({confidence_level*100}% confidence interval)',
                )
                for label in order
            ]

    if legend:
        ax.legend(
            bbox_to_anchor=(1, 1) if legend == 'out' else None,
            handles=handles,
            loc=patch_location,
            prop={'size': patch_size},
            borderaxespad=0.1,
        )

    return ax
    
def bot_bar_plot(
    data,
    palette=None,
    lrot=0,
    figsize=(5, 5),
    title='',
    ax=None,
    order=None,
    stars=False,
    percent=False,
    pvalue=False,
    p_digits=5,
    legend=True,
    xl=True,
    offset=-0.1,
    linewidth=0,
    align='center',
    bar_width=0.9,
    edgecolor=None,
    hide_grid=True,
    draw_horizontal=False,
    plot_all_borders=True,
    **kwargs
):
    """
    Plot a stacked bar plot based on contingency table

    Parameters
    ----------
    data: pd.DataFrame
        contingency table for plotting. Each element of index corresponds to a bar.
    palette: dict
        palette for plotting. Keys are unique values from groups, entries are color hexes
    lrot: float
        rotation angle of bar labels in degrees
    figsize: (float, float)
        figure size in inches
    title: str
        plot title
    ax: matplotlib axis
        axis to plot on
    order: list
        what order to plot the stacks of each bar in. Contains column labels of "data"
    stars: bool
        whether to use the star notation for p value instead of numerical value
    percent: bool
        whether to normalize each bar to 1
    pvalue: bool
        whether to add the p value (chi2 contingency test) to the plot title.
    p_digits: int
        number of digits to round the p value to
    legend: bool
        whether to plot the legend
    xl: bool
        whether to plot bar labels (on x axis for horizontal plot, on y axis for vertical plot)
    hide_grid: bool
        whether to hide grid on plot
    draw_horizontal: bool
        whether to draw horizontal bot bar plot
    plot_all_borders: bool
        whether to plot top and right border

    Returns
    -------
    matplotlib axis
    """
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if pvalue:
        from scipy.stats import chi2_contingency

        chi2_test_data = chi2_contingency(data)
        p = chi2_test_data[1]
        if title is not False:
            title += '\n' + get_pvalue_string(p, p_digits, stars=stars)

    if percent:
        c_data = data.apply(lambda x: x * 1.0 / x.sum(), axis=1)
        if title:
            title = '% ' + title
        ax.set_ylim(0, 1)
    else:
        c_data = data

    c_data.columns = [str(x) for x in c_data.columns]

    if order is None:
        order = c_data.columns
    else:
        order = [str(x) for x in order]

    if palette is None:
        c_palette = lin_colors(pd.Series(order))

        if len(order) == 1:
            c_palette = {order[0]: blue_color}
    else:
        c_palette = {str(k): v for k, v in palette.items()}

    if edgecolor is not None:
        edgecolor = [edgecolor] * len(c_data)

    kind_type = 'bar'
    if draw_horizontal:
        kind_type = 'barh'

    c_data[order].plot(
        kind=kind_type,
        stacked=True,
        position=offset,
        width=bar_width,
        color=pd.Series(order).map(c_palette).values,
        ax=ax,
        linewidth=linewidth,
        align=align,
        edgecolor=edgecolor,
    )

    ax = bot_bar_plot_prettify_axis(
        ax,
        c_data,
        legend,
        draw_horizontal,
        xl,
        lrot,
        title,
        hide_grid,
        plot_all_borders,
        **kwargs
    )

    if percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    return ax


def bot_bar_plot_prettify_axis(ax, c_data, legend, draw_horizontal, xl, lrot, title, hide_grid, plot_all_borders,
                               **kwargs):
    """
    Change some properties of bot_bar_plot ax

    Returns
    -------
    prettified axis
    """

    if legend:
        ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.1)
    else:
        ax.legend_.remove()

    if 'ylabel' in kwargs.keys():
        ax.set(ylabel=kwargs['ylabel'])

    if 'xlabel' in kwargs.keys():
        ax.set(xlabel=kwargs['xlabel'])

    if not draw_horizontal:
        ax.set_xticks(np.arange(len(c_data.index)) + 0.5)
        if xl:
            ax.set_xticklabels(c_data.index, rotation=lrot)
        else:
            ax.set_xticklabels([])
    else:
        ax.set_yticks(np.arange(len(c_data.index)) + 0.5)
        if xl:
            ax.set_yticklabels(c_data.index, rotation=lrot)
        else:
            ax.set_yticklabels([])

    if title is not False:
        ax.set_title(title)

    if hide_grid:
        ax.grid(False)

    sns.despine(ax=ax)

    if plot_all_borders:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    return ax

def axis_matras(ys, x_len, title, title_y=1):
    fig, axes = plt.subplots(nrows=len(ys), sharex=True, figsize=(x_len, np.sum(ys)), gridspec_kw={"height_ratios": ys})
    fig.suptitle(title, y=title_y)
    for i, ax in enumerate(axes):       
        yield ax

    plt.tight_layout()
    plt.subplots_adjust(top=2)
    
def line_palette_annotation_plot(series, palette, ax=None):
        for idx, category in enumerate(series):
                ax.axvline(x=idx, color=palette[category], linewidth=10)
                ax.set_yticklabels([])
                ax.set_xticklabels([])

def axis_net(x, y, title='', x_len=4, y_len=4, title_y=1, gridspec_kw=None):
    """
    Return an axis iterative for subplots arranged in a net
    :param x: int, number of subplots in a row
    :param y: int, number of subplots in a column
    :param title: str, plot title
    :param x_len: float, width of a subplot in inches
    :param y_len: float, height of a subplot in inches
    :param gridspec_kw: is used to specify axis ner with different rows/cols sizes.
            A dict: height_ratios -> list + width_ratios -> list
    :param title_y: absolute y position for suptitle
    :return: axs.flat, numpy.flatiter object which consists of axes (for further plots)
    """
    if x == y == 1:
        fig, ax = plt.subplots(figsize=(x * x_len, y * y_len))
        af = ax
    else:
        fig, axs = plt.subplots(y, x, figsize=(x * x_len, y * y_len), gridspec_kw=gridspec_kw)
        af = axs.flat

    fig.suptitle(title, y=title_y)
    return af


def lin_colors(factors_vector, cmap='default', sort=True, min_v=0, max_v=1, linspace=True):
    """
    Return dictionary of unique features of "factors_vector" as keys and color hexes as entries
    :param factors_vector: pd.Series
    :param cmap: matplotlib.colors.LinearSegmentedColormap, which colormap to base the returned dictionary on
        default - matplotlib.cmap.hsv with min_v=0, max_v=.8, lighten_color=.9
    :param sort: bool, whether to sort the unique features
    :param min_v: float, for continuous palette - minimum number to choose colors from
    :param max_v: float, for continuous palette - maximum number to choose colors from
    :param linspace: bool, whether to spread the colors from "min_v" to "max_v"
        linspace=False can be used only in discrete cmaps
    :return: dict
    """

    unique_factors = factors_vector.dropna().unique()
    if sort:
        unique_factors = np.sort(unique_factors)

    if cmap == 'default':
        cmap = matplotlib.cm.rainbow
        max_v = .92

    if linspace:
        cmap_colors = cmap(np.linspace(min_v, max_v, len(unique_factors)))
    else:
        cmap_colors = np.array(cmap.colors[:len(unique_factors)])

    return dict(list(zip(unique_factors, [matplotlib.colors.to_hex(x) for x in cmap_colors])))


def axis_net(x, y, title='', x_len=4, y_len=4, title_y=1, gridspec_kw=None):
    """
    Return an axis iterative for subplots arranged in a net
    :param x: int, number of subplots in a row
    :param y: int, number of subplots in a column
    :param title: str, plot title
    :param x_len: float, width of a subplot in inches
    :param y_len: float, height of a subplot in inches
    :param gridspec_kw: is used to specify axis ner with different rows/cols sizes.
            A dict: height_ratios -> list + width_ratios -> list
    :param title_y: absolute y position for suptitle
    :return: axs.flat, numpy.flatiter object which consists of axes (for further plots)
    """
    if x == y == 1:
        fig, ax = plt.subplots(figsize=(x * x_len, y * y_len))
        af = ax
    else:
        fig, axs = plt.subplots(y, x, figsize=(x * x_len, y * y_len), gridspec_kw=gridspec_kw)
        af = axs.flat

    fig.suptitle(title, y=title_y)
    return af



def matrix_projection_plot(
    data: pd.DataFrame,
    grouping: Optional[pd.Series] = None,
    p_model: Literal['PCA', 'UMAP', 'TSNE'] = 'PCA',
    order: Optional[List[Any]] = None,
    n_components: int = 2,
    ax: Optional[matplotlib.axes.Axes] = None,
    palette: Optional[Dict[Any, str]] = None,
    confidence_level: bool = False,
    centroids: bool = False,
    centroid_complement_color: bool = False,
    random_state: int = 42,
    figsize: Tuple[float, float] = (5, 5),
    title: str = '',
    return_model: bool = False,
    legend: Union[str, None] = 'out',
    plot_limits: bool = False,
    label_samples: bool = False,
    kwargs_scatter: Optional[Dict[Any, Any]] = None,
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Performs a data dimensionality reduction using p_model
    and then plots as a scatter plot colored by grouping.
    If n_components > 2, then first two components will be chosen for plotting.

    Usually the function is not called directly,
    but is called via pca_plot(), tsne_plot() ot umap_plot().

    :param data: Samples as indexes and features as columns.
    :param grouping: Series of samples to group correspondence.
    :param p_model: {'PCA', 'TSNE', 'UMAP'}, default: 'PCA'
        The model that will be used for dimensionality reduction.
    :param order: Groups plotting order (useful for limited groups displaying)
    :param n_components: int, default: 2 The number of dimensions to reduce the data to.
    :param ax: axes to plot
    :param palette: Colors corresponding to the groups. If None -> lin_colors will be applied.
    :param confidence_level: {80, 90, 95, 99} or False, defaut: False
        Confidence level, based on pearson correlation and standard deviation
    :param return_model: bool, default: False
        If True -> return Tuple[ax, transformed_data, model]
    :param alpha: plotting option
    :param random_state: 42
    :param s: point size
    :param figsize: if ax=None a new axis with the figsize will be created
    :param title: plot title
    :param legend: {'in', 'out'} or None, default: 'in'
        'in' - plots the legend inside the plot, 'out' - outside. Otherwise - no legend
    :param plot_limits: limits axes size respect to a plot
    :param label_samples: bool, default: False
        Whether to subscribe samples' names on plot.
    :param kwargs_scatter: dict
        Dict with various params for ax.scatter - marker, linewidth, edgecolor.
    :param kwargs: kwargs for projection model, n_jobs is set to 4 by default for UMAP and TSNE
    :return: matplotlib.axes.Axes
    """
    kwargs.setdefault('n_jobs', 4)

    if grouping is None:
        grouping = item_series('*', data)

    # Common samples
    c_data, c_grouping = to_common_samples([data, grouping])

    if order:
        group_order = order
    else:
        group_order = np.sort(c_grouping.unique())

    if palette is None:
        cur_palette = lin_colors(c_grouping)
    else:
        cur_palette = palette

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Get model and transform
    n_components = min(n_components, len(c_data.columns))
    if p_model == 'PCA':
        from sklearn.decomposition import PCA

        kwargs.pop('n_jobs', 'None')  # PCA does not support n_jobs
        model = PCA(n_components=n_components, random_state=random_state, **kwargs)

    elif p_model == 'TSNE':
        from sklearn.manifold import TSNE

        model = TSNE(
            n_components=n_components, random_state=random_state, **kwargs
        )

        label_1 = 'tSNE 1'
        label_2 = 'tSNE 2'
    elif p_model == 'UMAP':
        from umap import UMAP

        model = UMAP(n_components=n_components, random_state=random_state, **kwargs)

        label_1 = 'UMAP 1'
        label_2 = 'UMAP 2'
    else:
        raise Exception('Unknown model')

    data_tr = pd.DataFrame(model.fit_transform(c_data), index=c_data.index)

    if p_model == 'PCA':
        label_1 = 'PCA 1 component {}% variance explained'.format(
            int(model.explained_variance_ratio_[0] * 100)
        )
        label_2 = 'PCA 2 component {}% variance explained'.format(
            int(model.explained_variance_ratio_[1] * 100)
        )

    kwargs_scatter = kwargs_scatter or {}
    simple_palette_scatter(
        x=data_tr[0],
        y=data_tr[1],
        grouping=c_grouping,
        order=group_order,
        palette=cur_palette,
        ax=ax,
        legend=legend,
        confidence_level=confidence_level,
        centroids=centroids,
        centroid_complement_color=centroid_complement_color,
        **kwargs_scatter,
    )

    if plot_limits:
        x_lim_min = data_tr[0].min()
        x_lim_max = data_tr[0].max()
        y_lim_min = data_tr[1].min()
        y_lim_max = data_tr[1].max()
        if label_samples:
            delta_x = (x_lim_max + x_lim_min) / 2
            delta_y = (y_lim_max + y_lim_min) / 2
        else:
            delta_x = (x_lim_max + x_lim_min) / 20
            delta_y = (y_lim_max + y_lim_min) / 20

        ax.set_xlim([x_lim_min - delta_x, x_lim_max + delta_x])
        ax.set_ylim([y_lim_min - delta_y, y_lim_max + delta_x])

    if label_samples:
        texts = []
        sample_names = list(c_grouping.index)
        X = list(data_tr[0])
        Y = list(data_tr[1])
        for i, name in enumerate(sample_names):
            texts.append(plt.text(X[i], Y[i], s=str(name), fontsize=8))
        adjust_text(
            texts, x=X, y=Y, arrowprops=dict(arrowstyle='-', color='black', lw=0.1)
        )

    ax.set_title(title)
    ax.set_xlabel(label_1)
    ax.set_ylabel(label_2)

    if return_model:
        return ax, data_tr, model
    return ax


def pca_plot(
    data: pd.DataFrame,
    grouping: Optional[pd.Series] = None,
    n_components: int = 2,
    title: str = '',
    ax: Optional[matplotlib.axes.Axes] = None,
    order: Optional[List[Any]] = None,
    palette: Optional[Dict[Any, str]] = None,
    confidence_level: bool = False,
    centroids: bool = False,
    centroid_complement_color: bool = False,
    return_model: bool = False,
    legend: Union[str, None] = 'out',
    **kwargs,
) -> matplotlib.axes.Axes:
    """
    Perform dimensionality reduction using Principal Component Analysis
    and plot results as scatter plot.

    See Also
    ------------------
    plotting.matrix_projection_plot

    :param data: pd.DataFrame
        Samples as indexes and features as columns.
    :param grouping: pd.Series
        Series of samples to group correspondence.
    :param n_components: int, default: 2
        The number of components, that will be calculated.
    :param title: str, title of plot
    :param ax: axes to plot
    :param order: list
        Groups plotting order (useful for limited groups displaying)
    :param palette: dict
        Colors corresponding to the groups.
        If None -> lin_colors will be applied.
    :param confidence_level: {80, 90, 95, 99} or False, defaut: False
        Confidence level, based on pearson correlation and standard deviation
    :param return_model: bool, default: False
        If True -> return Tuple[ax, transformed_data, model]
    :param legend: {'in', 'out'} or None, default: 'in'
        'in' - plots the legend inside the plot, 'out' - outside. Otherwise - no legend
    :param kwargs:
    :return: matplotlib.axes.Axes

    Example
    ------------------------
    # If we want to get four components from PCA
    ax, transformed_data, model = pca_plot(data=expressions, grouping=subtype, return_model=True, n_components=4)

    # If we want to annotate outliers
    pca_plot(data=expressions, grouping=subtype, label_samples=True)
    """

    kwargs_scatter = dict()
    kwargs_scatter['linewidth'] = kwargs.pop('linewidth', 0)
    kwargs_scatter['marker'] = kwargs.pop('marker', 'o')
    kwargs_scatter['edgecolor'] = kwargs.pop('edgecolor', 'black')
    kwargs_scatter['s'] = kwargs.pop('s', 20)
    kwargs_scatter['alpha'] = kwargs.pop('alpha', 1)

    return matrix_projection_plot(
        data=data,
        grouping=grouping,
        p_model='PCA',
        n_components=n_components,
        title=title,
        ax=ax,
        order=order,
        palette=palette,
        return_model=return_model,
        legend=legend,
        confidence_level=confidence_level,
        centroids=centroids,
        centroid_complement_color=centroid_complement_color,
        kwargs_scatter=kwargs_scatter,
        **kwargs,
    )


def tsne_plot(
    data,
    grouping=None,
    n_components=2,
    title='',
    ax=None,
    order=(),
    palette=None,
    return_model=False,
    legend='in',
    **kwargs,
):
    kwargs['perplexity'] = kwargs.get('perplexity', 30)

    kwargs_scatter = dict()
    kwargs_scatter['linewidth'] = kwargs.pop('linewidth', 0)
    kwargs_scatter['marker'] = kwargs.pop('marker', 'o')
    kwargs_scatter['edgecolor'] = kwargs.pop('edgecolor', 'black')

    return matrix_projection_plot(
        data,
        grouping,
        p_model='TSNE',
        n_components=n_components,
        title=title,
        ax=ax,
        order=order,
        palette=palette,
        return_model=return_model,
        legend=legend,
        kwargs_scatter=kwargs_scatter,
        **kwargs,
    )


def umap_plot(
    data,
    grouping=None,
    n_components=2,
    title='',
    ax=None,
    order=(),
    palette=None,
    return_model=False,
    legend='in',
    **kwargs,
):
    kwargs['n_neighbors'] = kwargs.get('n_neighbors', 15)
    kwargs['min_dist'] = kwargs.get('min_dist', 0.1)
    kwargs['metric'] = kwargs.get('metric', 'euclidean')

    kwargs_scatter = dict()
    kwargs_scatter['linewidth'] = kwargs.pop('linewidth', 0)
    kwargs_scatter['marker'] = kwargs.pop('marker', 'o')
    kwargs_scatter['edgecolor'] = kwargs.pop('edgecolor', 'black')
    kwargs_scatter['s'] = kwargs.pop('s', 20)
    kwargs_scatter['alpha'] = kwargs.pop('alpha', 1)

    return matrix_projection_plot(
        data,
        grouping,
        p_model='UMAP',
        n_components=n_components,
        title=title,
        ax=ax,
        order=order,
        palette=palette,
        return_model=return_model,
        legend=legend,
        kwargs_scatter=kwargs_scatter,
        **kwargs,
    )

def clustering_heatmap(ds, title='', corr='pearson', method='complete',
                       yl=True, xl=True,
                       cmap=matplotlib.cm.coolwarm, col_colors=None,
                       figsize=None, **kwargs):
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage

    dissimilarity_matrix = 1 - ds.T.corr(method=corr)
    hclust_linkage = linkage(squareform(dissimilarity_matrix), method=method)

    g = sns.clustermap(1 - dissimilarity_matrix, method=method,
                       row_linkage=hclust_linkage, col_linkage=hclust_linkage,
                       cmap=cmap, yticklabels=yl, xticklabels=xl,
                       col_colors=col_colors, figsize=figsize, **kwargs)

    g.fig.suptitle(title)

    return g


def patch_plot(patches, ax=None, order='sort', w=0.25, h=0, legend_right=True,
               show_ticks=False):
    cur_patches = pd.Series(patches)

    if order == 'sort':
        order = list(np.sort(cur_patches.index))

    data = pd.Series([1] * len(order), index=order[::-1])
    if ax is None:
        if h == 0:
            h = 0.3 * len(patches)
        _, ax = plt.subplots(figsize=(w, h))

    data.plot(kind='barh', color=[cur_patches[x] for x in data.index], width=1, ax=ax)
    ax.set_xticks([])
    if legend_right:
        ax.yaxis.tick_right()

    sns.despine(offset={'left': -2}, ax=ax)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not show_ticks:
        ax.tick_params(length=0)

    return ax



def draw_graph(G, ax=None, title='', figsize=(12, 12), v_labels=True, e_labels=True, node_color='r', node_size=30,
               el_fs=5, nl_fs=8):
    """
    Draws a graph.
    :param G:
    :param ax:
    :param title:
    :param figsize:
    :param v_labels:
    :param e_labels:
    :param node_color:
    :param node_size:
    :param el_fs: edge label font size
    :param nl_fs: node label font size
    :return:
    """
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    if v_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=nl_fs, font_family='sans-serif', font_color='blue')

    nx.draw_networkx_edges(G, pos, ax=ax)
    if e_labels:
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=el_fs, width=labels, edge_labels=labels)

    ax.set_title(title, fontsize=18)
    return ax

def calculate_and_plot_correlations(series1, series2,name1='Predicted', name2='True',
                                    verbose=True, ret=False, plot=True,title=False):   
    """
    Calculate and plot various correlation metrics between two series and optionally return these metrics.

    This function computes Pearson and Spearman correlations, mean squared error (MSE), mean absolute error (MAE), and concordance correlation coefficient (CCC) between two provided series. It also supports plotting these correlations along with a linear regression line and marginal histograms.

    :param series1: pd.Series
        The first series for comparison.
    :param series2: pd.Series
        The second series for comparison.
    :param name1: str, default 'Predicted'
        Label for series1 in the plot.
    :param name2: str, default 'True'
        Label for series2 in the plot.
    :param verbose: bool, default True
        If True, prints the computed metrics.
    :param ret: bool, default False
        If True, returns a dictionary of computed metrics.
    :param plot: bool, default True
        If True, generates a joint plot showing the correlation.
    :param title: Optional[str]
        Title for the plot. If None, no title is set.

    :return: Union[None, Dict[str, float]]
        If ret is True, returns a dictionary with 'MSE', 'MAE', 'Spearman', 'CCC', and 'Pearson' as keys.
        Otherwise, returns None.

    Example
    ------------------------
    # Plot and get correlation metrics
    metrics = calculate_and_plot_correlations(series1, series2, ret=True)

    # Just plot the correlations
    calculate_and_plot_correlations(series1, series2, plot=True, verbose=True)
    """                                   
    from sklearn import metrics
    series1, series2 = to_common_samples((series1.dropna(), series2.dropna()))
    df = pd.DataFrame({name1: series1, name2: series2})
    pearson_corr, pearson_p = stats.pearsonr(series1, series2)
    spearman_corr, spearman_p = stats.spearmanr(series1, series2)
    mse = metrics.mean_squared_error(series1, series2)
    mae = metrics.mean_absolute_error(series1, series2)
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(series1, series2)
        concordance_corr_coef = (2 * r_value * np.std(series1) * np.std(series2)) / (np.var(series1) + np.var(series2) + (np.mean(series1) - np.mean(series2))**2)
    except:
        concordance_corr_coef = 0
    text =  [f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.1e}",
          f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.1e}",
    f"MSE: {mse:.4f}; MAE: {mae:.4f}",
    f"CCC: {concordance_corr_coef:.4f}",
    f"Number of samples: {df.dropna().shape[0]}"]
    if verbose:
        print('\n'.join(text))
    if plot:
        g = sns.JointGrid(data=df, x=name1, y=name2, space=0)
        g.plot_joint(sns.scatterplot, alpha=0.6,color = 'teal')
        sns.regplot(x=name1, y=name2, data=df, ax=g.ax_joint, scatter=False, color='r')
        g.plot_marginals(sns.histplot, kde=True, color="teal")
        if title:
            text+=[title]
        textstr = '\n'.join(text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        g.ax_joint.text(0.05, 0.95, textstr, transform=g.ax_joint.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, linespacing=1.5)
        plt.show()
        plt.close()
    if ret:
        metrics = {
        'MSE': mse,
        'MAE': mae,
        'Spearman':spearman_corr,
        'CCC':concordance_corr_coef,
        'Pearson':pearson_corr}
        return metrics


def define_ax_figsize(ax):
    """
    Function calculates figsize for given ax.
    Calculations are quite tricky when ax come from figure with multiple subplots

    :param ax: amatplotlib axis
    :return: (float, float), calculated figure size
    """
    full_fig_size = list(ax.figure.get_size_inches())
    subplots_num = ax.get_subplotspec().get_geometry()[:2]
    current_ax_num = ax.get_subplotspec().get_geometry()[2]

    current_ax_x = current_ax_num % subplots_num[1]
    current_ax_y = current_ax_num // subplots_num[1]

    height_ratios = ax.get_subplotspec().get_gridspec().get_height_ratios()
    width_ratios = ax.get_subplotspec().get_gridspec().get_width_ratios()

    if height_ratios is None:
        # in this case all ratios are equal
        height_ratios = [1] * subplots_num[0]

    if width_ratios is None:
        # in this case all ratios are equal
        width_ratios = [1] * subplots_num[1]

    return (
        width_ratios[current_ax_x] / float(sum(width_ratios)) * full_fig_size[0],
        height_ratios[current_ax_y] / float(sum(height_ratios)) * full_fig_size[1],
    )

def boxplot_with_pvalue(
    data,
    grouping,
    title='',
    ax=None,
    figsize=None,
    swarm=True,
    p_digits=3,
    stars=True,
    violin=False,
    palette=None,
    order=None,
    y_min=None,
    y_max=None,
    s=7,
    p_fontsize=16,
    xlabel=None,
    **kwargs,
):
    """
    Plots boxplot or violin plot with pairwise comparisons
    :param data: pd.Series, series with numerical data
    :param grouping: pd.Series, series with categorical data
    :param title: str, plot title
    :param ax: matplotlib axis, axis to plot on
    :param figsize: (float, float), figure size in inches
    :param swarm: bool, whether to plot a swarm in addition to boxes
    :param p_digits: int, number of digits to round p value to
    :param stars: bool, whether to plot star notation instead of number for p value
    :param violin: bool, whether to do a violin plot
    :param palette: dict, palette for plotting. Keys are unique values from groups, entries are color hexes
    :param order: list, order to plot the entries in. Contains ordered unique values from "grouping"
    :param y_min: float, vertical axis minimum
    :param y_max:float, vertical axis maximum
    :param s: float, size of dots in swarmplot
    :param p_fontsize: float, font size for p value labels
    :param kwargs:
    :return: matplotlib axis
    """

    from scipy.stats import mannwhitneyu

    if data.index.duplicated().any() | grouping.index.duplicated().any():
        raise Exception('Indexes contain duplicates')

    cdata, cgrouping = to_common_samples([data.dropna(), grouping.dropna()])

    if len(cgrouping.dropna().unique()) < 2:
        raise Exception(
            'Less from 2 classes provided: {}'.format(len(cgrouping.unique()))
        )

    if order is None:
        order = cgrouping.dropna().unique()

    if ax is None:
        if figsize is None:
            figsize = (1.2 * len(order), 4)
        _, ax = plt.subplots(figsize=figsize)

    if not violin:
        sns.boxplot(
            y=cdata,
            x=cgrouping,
            ax=ax,
            palette=palette,
            order=order,
            fliersize=0,
            **kwargs,
        )
    else:
        sns.violinplot(
            y=cdata, x=cgrouping, ax=ax, palette=palette, order=order, **kwargs
        )

        # Ignoring swarm setting since violin performs same function
        swarm = False

    if swarm:
        sns.swarmplot(y=cdata, x=cgrouping, ax=ax, color=".25", order=order, s=s)

    pvalues = []
    for g1, g2 in zip(order[:-1], order[1:]):
        samples_g1 = cgrouping[cgrouping == g1].index
        samples_g2 = cgrouping[cgrouping == g2].index
        try:
            if len(samples_g1) and len(samples_g2):
                pv = mannwhitneyu(
                    cdata.loc[samples_g1],
                    cdata.loc[samples_g2],
                    alternative='two-sided',
                ).pvalue
            else:
                pv = 1
        except ValueError:
            pv = 1
        pvalues.append(pv)

    y_max = y_max or max(cdata)
    y_min = y_min or min(cdata)
    effective_size = y_max - y_min
    plot_y_limits = (y_min - effective_size * 0.15, y_max + effective_size * 0.2)

    if p_digits > 0:

        pvalue_line_y_1 = y_max + effective_size * 0.05
        if figsize is None:
            figsize = define_ax_figsize(ax)
        pvalue_text_y_1 = pvalue_line_y_1 + 0.25 * effective_size / figsize[1]

        for pos, pv in enumerate(pvalues):
            pvalue_str = get_pvalue_string(pv, p_digits, stars=stars)
            pvalue_text_y_1_local = pvalue_text_y_1

            if pvalue_str == '-':
                pvalue_text_y_1_local += 0.1 * effective_size / figsize[1]

            bar_fraction = str(0.25 / 2.0 / (figsize[0] / float(len(order))))

            ax.annotate(
                "",
                xy=(pos + 0.1, pvalue_line_y_1),
                xycoords='data',
                xytext=(pos + 0.9, pvalue_line_y_1),
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="-",
                    ec='#000000',
                    connectionstyle="bar,fraction={}".format(bar_fraction),
                ),
            )
            ax.text(
                pos + 0.5,
                pvalue_text_y_1_local,
                pvalue_str,
                fontsize=p_fontsize,
                horizontalalignment='center',
                verticalalignment='center',
            )

    ax.set_title(title)
    ax.set_ylim(plot_y_limits)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    return ax