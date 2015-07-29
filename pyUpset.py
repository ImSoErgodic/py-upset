__author__ = 'leo@opensignal.com'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations


def plot(data_dict, *, unique_keys=None, sort_by='size', inters_size_bounds=(0, np.inf),
         inters_degree_bounds=(1, np.inf), additional_plots=None, query=None):
    """
    Plots a main set of graph showing intersection size, intersection matrix and the size of base sets. If given,
    additional plots are placed below the main graph.

    :param data_dict: dictionary like {data_frame_name: data_frame}

    :param unique_keys: list. Specifies the names of the columns that, together, can uniquely identify a row. If left
    empty, pyUpSet will try to use all common columns in the data frames and may possibly raise an exception (no
    common columns) or produce unexpected results (columns in different data frames with same name but different
    meanings/data).

    :param sort_by: 'size' or 'degree'. The order in which to sort the intersection bar chart and matrix in the main
    graph

    :param inters_size_bounds: tuple. Specifies the size limits of the intersections that will be displayed.
    Intersections (and relative data) whose size is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param inters_degree_bounds: tuple. Specified the degree limits of the intersections that will be displayed.
    Intersections (and relative data) whose degree is outside the interval will not be plotted. Defaults to (0, np.inf).

    :param additional_plots: list of dictionaries. See below for details.

    :param query: list of tuples. See below for details.

    :return: dictionary of matplotlib objects, namely the figure and the axes.

    :raise ValueError: if no unique_keys are specified and the data frames have no common column names.

    The syntax to specify additional plots follows the signature of the corresponding matplotlib method in an Axes
    class. For each additional plot one specifies a dictionary with the kind of plot, the columns name to retrieve
    relevant data and the kwargs to pass to the plot function, as in `{'kind':'scatter', 'data':{'x':'col_1',
    'y':'col_2'}, 'kwargs':{'s':50}}`.

    It is also possible to highlight intersections. This is done through the `query` argument, where the
    intersections to highligh must be specified with the names used as keys in the data_dict.

    """
    ap = [] if additional_plots is None else additional_plots
    all_columns = unique_keys if unique_keys is not None else __get_all_common_columns(data_dict)
    all_columns = list(all_columns)

    plot_data = DataExtractor(data_dict, all_columns)
    ordered_inters_sizes, ordered_in_sets, ordered_out_sets = plot_data.get_filtered_intersections(sort_by,
                                                                                                   inters_size_bounds,
                                                                                                   inters_degree_bounds)
    ordered_dfs, ordered_df_names = plot_data.ordered_dfs, plot_data.ordered_df_names

    upset = UpSetPlot(len(ordered_dfs), len(ordered_in_sets), additional_plots, query)
    fig_dict = upset.main_plot(ordered_dfs, ordered_df_names, ordered_in_sets, ordered_out_sets,
                                  ordered_inters_sizes)
    fig_dict['additional'] = []

    for i, pl in enumerate(ap):
        plot_kind = pl['kind']
        data_values = plot_data.extract_data_for[plot_kind](**pl['data'])
        graph_kwargs = pl['graph_kwargs'] if pl.__contains__('graph_kwargs') else {}
        pm = upset._plot_method[plot_kind]
        ax = pm(i, data_values, graph_kwargs)
        fig_dict['additional'].append(ax)

    return fig_dict


def __get_all_common_columns(data_dict):
    """
    Computes an array of (unique) common columns to the data frames in data_dict
    :param data_dict: Dictionary of data frames
    :return: array.
    """
    common_columns = []
    for i, k in enumerate(data_dict.keys()):
        if i == 0:
            common_columns = data_dict[k].columns
        else:
            common_columns = common_columns.intersection(data_dict[k].columns)
    if common_columns.values:
        raise ValueError('Data frames should have homogeneous columns with the same name to use for computing '
                         'intersections')
    return common_columns.unique()


class UpSetPlot():
    def __init__(self, rows, cols, additional_plots, query):
        """
        Class linked to a data set; it contains the methods to produce plots according to the UpSet representation.
        :type plot_data: DataExtractor
        :param plot_data: PlotData object
        """

        # set standard colors
        self.greys = plt.cm.Greys([.22, .8])

        # map of additional plot names to internal methods
        self._plot_method = {
            'scatter': self._scatter
        }

        # map queries to graphic properties
        self.query = [] if query is None else query
        qu_col = plt.cm.rainbow(np.linspace(.01, .99, len(self.query)))
        self.query2color = dict(zip([frozenset(q) for q in self.query], qu_col))
        self.query2zorder = dict(zip([frozenset(q) for q in self.query], np.arange(len(self.query)) + 1))

        # set figure properties
        self.rows = rows
        self.cols = cols
        self.x_values, self.y_values = self._create_coordinates(rows, cols)
        self.fig, self.ax_intbars, self.ax_intmatrix, \
        self.ax_setsize, self.add_plots_axes = self._prepare_figure(additional_plots)

        # single dictionary may be fragile - I leave it here as a future option
        # self.query2kwargs = dict(zip([frozenset(q) for q in self.query],
        # [dict(zip(['color', 'zorder'],
        # [col, 1])) for col in qu_col]))

    def _create_coordinates(self, rows, cols):
        x_values = (np.arange(cols) + 1)
        y_values = (np.arange(rows) + 1)
        return x_values, y_values

    def _prepare_figure(self, additional_plots):
        fig = plt.figure(figsize=(16, 10))
        if additional_plots:
            main_gs = gridspec.GridSpec(3, 1, hspace=.6)
            topgs = main_gs[:2, 0]
            botgs = main_gs[2, 0]
        else:
            topgs = gridspec.GridSpec(1, 1)[0, 0]
        fig_cols = self.cols + 3
        fig_rows = self.rows + self.rows * 4

        gs_top = gridspec.GridSpecFromSubplotSpec(fig_rows, fig_cols, subplot_spec=topgs, wspace=.1, hspace=.2)
        setsize_w, setsize_h = 3, self.rows
        intmatrix_w, intmatrix_h = setsize_w + self.cols, self.rows
        intbars_w, intbars_h = setsize_w + self.cols, self.rows * 4
        ax_setsize = plt.subplot(gs_top[-1:-setsize_h, 0:setsize_w])
        ax_intmatrix = plt.subplot(gs_top[-1:-intmatrix_h, setsize_w:intmatrix_w])
        ax_intbars = plt.subplot(gs_top[:self.rows * 4 - 1, setsize_w:intbars_w])

        add_ax = []
        if additional_plots:
            num_plots = len(additional_plots)
            num_bot_rows, num_bot_cols = int(np.ceil(num_plots / 2)), 2
            gs_bottom = gridspec.GridSpecFromSubplotSpec(num_bot_rows, num_bot_cols,
                                                         subplot_spec=botgs, wspace=.15, hspace=.2)
            from itertools import product

            for r, c in product(range(num_bot_rows), range(num_bot_cols)):
                new_plotL = plt.subplot(gs_bottom[r, c])
                add_ax.append(new_plotL)

        return fig, ax_intbars, ax_intmatrix, ax_setsize, tuple(add_ax)

    def _color_for_query(self, query):
        query_color = self.query2color.setdefault(query, self.greys[1])
        return query_color

    def _zorder_for_query(self, query):
        query_zorder = self.query2zorder.setdefault(query, 0)
        return query_zorder

    def main_plot(self, ordered_dfs, ordered_df_names, ordered_in_sets, ordered_out_sets, ordered_inters_sizes):
        ylim = self._base_sets_plot(ordered_dfs, ordered_df_names)
        xlim = self._inters_sizes_plot(ordered_in_sets, ordered_inters_sizes)
        set_row_map = dict(zip(ordered_df_names, self.y_values))
        self._inters_matrix(ordered_in_sets, ordered_out_sets, xlim, ylim, set_row_map)
        return {'figure': self.fig,
                'intersection_bars':self.ax_intbars,
                'intersection_matrix':self.ax_intmatrix,
                'base_set_size':self.ax_setsize}

    def _base_sets_plot(self, sorted_sets, sorted_set_names):
        ax = self.ax_setsize
        ax.invert_xaxis()
        height = .6
        bar_bottoms = self.y_values - height / 2

        ax.barh(bar_bottoms, [len(x) for x in sorted_sets], height=height, color=self.greys[1])

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))

        self._strip_axes(ax, keep_spines=['bottom'], keep_ticklabels=['bottom'])

        ax.set_ylim((height / 2, ax.get_ylim()[1] + height / 2))
        xlim = ax.get_xlim()
        gap = max(xlim) / 500.0 * 20
        ax.set_xlim(xlim[0] + gap, xlim[1] - gap)
        xlim = ax.get_xlim()
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

        for i, (x, y) in enumerate(zip([len(x) for x in sorted_sets], self.y_values)):
            ax.annotate(sorted_set_names[i], rotation=90, ha='right', va='bottom', fontsize=15,
                        xy=(x, y), xycoords='data',
                        xytext=(-30, 0), textcoords='offset points',
                        arrowprops=dict(arrowstyle="-[",
                                        shrinkA=1,
                                        shrinkB=3,
                                        connectionstyle='arc,angleA=-180, angleB=180, armB=30'  # widthB to control
                                        # bracket
                                        ),
                        )

        ax.set_xlabel("Set size", fontweight='bold', fontsize=13)

        return ax.get_ylim()

    def _strip_axes(self, ax, keep_spines=None, keep_ticklabels=None):
        tick_params_dict = {'which': 'both',
                            'bottom': 'off',
                            'top': 'off',
                            'left': 'off',
                            'right': 'off',
                            'labelbottom': 'off',
                            'labeltop': 'off',
                            'labelleft': 'off',
                            'labelright': 'off'}
        if keep_ticklabels is None:
            keep_ticklabels = []
        if keep_spines is None:
            keep_spines = []
        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
        for k in lab_keys:
            tick_params_dict[k[0]] = 'on'
            tick_params_dict[k[1]] = 'on'
        ax.tick_params(**tick_params_dict)
        for sname, spine in ax.spines.items():
            if sname not in keep_spines:
                spine.set_visible(False)

    def _inters_sizes_plot(self, ordered_in_sets, inters_sizes):
        ax = self.ax_intbars
        width = .5
        self._strip_axes(ax, keep_spines=['left'], keep_ticklabels=['left'])

        bar_bottom_left = self.x_values - width / 2

        bar_colors = [self._color_for_query(frozenset(inter)) for inter in ordered_in_sets]

        ax.bar(bar_bottom_left, inters_sizes, width=width, color=bar_colors)

        ylim = ax.get_ylim()
        label_vertical_gap = (ylim[1] - ylim[0]) / 60

        for x, y in zip(self.x_values, inters_sizes):
            ax.text(x, y + label_vertical_gap, "%.2g" % y, rotation=90, ha='center', va='bottom')

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

        gap = max(ylim) / 500.0 * 20
        ax.set_ylim(ylim[0] - gap, ylim[1] + gap)
        ylim = ax.get_ylim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])

        ax.yaxis.grid(True, lw=.25, color='grey', ls=':')
        ax.set_ylabel("Intersection size", labelpad=6, fontweight='bold', fontsize=13)

        return ax.get_xlim()

    def _inters_matrix(self, ordered_in_sets, ordered_out_sets, xlims, ylims, set_row_map):
        ax = self.ax_intmatrix
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        row_width = self.x_values[1] - self.x_values[0]

        self._strip_axes(ax)

        background = plt.cm.Greys([.09])[0]

        from matplotlib.patches import Rectangle, Circle

        for r, y in enumerate(self.y_values):
            if r % 2 == 0:
                ax.add_patch(Rectangle((xlims[0], y - row_width / 2), height=row_width,
                                       width=xlims[1],
                                       color=background, zorder=0))

        for col_num, (in_sets, out_sets) in enumerate(zip(ordered_in_sets, ordered_out_sets)):
            in_y = [set_row_map[s] for s in in_sets]
            out_y = [set_row_map[s] for s in out_sets]
            # in_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[1]) for y in in_y]
            # out_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[0]) for y in out_y]
            # for c in chain.from_iterable([in_circles, out_circles]):
            # ax.add_patch(c)
            ax.scatter(np.repeat(self.x_values[col_num], len(in_y)), in_y, color=self._color_for_query(frozenset(
                in_sets)), s=300)
            ax.scatter(np.repeat(self.x_values[col_num], len(out_y)), out_y, color=self.greys[0], s=300)
            ax.vlines(self.x_values[col_num], min(in_y), max(in_y), lw=3.5, color=self._color_for_query(frozenset(
                in_sets)))

    def _scatter(self, ax_index, data_values, plot_kwargs):
        ax = self.add_plots_axes[ax_index]

        for data_item in data_values:
            ax.scatter(x=data_item['x'], y=data_item['y'],
                       color=self._color_for_query(frozenset(data_item['in_sets'])),
                       alpha=.3,
                       zorder=self._zorder_for_query(frozenset(data_item['in_sets'])),
                       **plot_kwargs)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

        self._strip_axes(ax, keep_spines=['bottom', 'left'], keep_ticklabels=['bottom', 'left'])
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        gap_y, gap_x = max(ylim) / 500.0 * 20, max(xlim) / 500.0 * 20
        ax.set_ylim(ylim[0] - gap_y, ylim[1] + gap_y)
        ax.set_xlim(xlim[0] - gap_x, xlim[1] + gap_x)
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

        return ax


class DataExtractor:
    def __init__(self, data_dict, columns):
        self.columns = columns
        self.ordered_dfs, self.ordered_df_names, self.df_dict = self.extract_base_sets_data(data_dict,
                                                                                            columns)
        self.in_sets_list, self.inters_degrees, \
        self.out_sets_list, self.inters_df_dict = self.extract_intersection_data()
        self.extract_data_for = {
            'scatter': self.__extract_data_for_scatter
        }


    def extract_base_sets_data(self, data_dict, columns):
        dfs = []
        df_names = []
        # extract interesting columns from dfs
        for name, df in data_dict.items():
            df_names.append(name)
            dfs.append(df[columns])
        df_names = np.array(df_names)
        # order dfs
        base_sets_order = np.argsort([x.shape[0] for x in dfs])[::-1]
        ordered_base_set_names = df_names[base_sets_order]
        ordered_base_sets = [data_dict[name] for name in ordered_base_set_names]
        set_dict = dict(zip(ordered_base_set_names, ordered_base_sets))

        return ordered_base_sets, ordered_base_set_names, set_dict

    def extract_intersection_data(self):
        in_sets_list = []
        out_sets_list = []
        inters_dict = {}
        inters_degrees = []
        for col_num, in_sets in enumerate(chain.from_iterable(
                combinations(self.ordered_df_names, i) for i in np.arange(1, len(self.ordered_dfs) + 1))):

            inters_degrees.append(len(in_sets))
            in_sets_list.append(in_sets)
            out_sets = set(self.ordered_df_names).difference(set(in_sets))
            in_sets_l = list(in_sets)
            out_sets_list.append(set(out_sets))

            seed = in_sets_l.pop()
            exclusive_intersection = pd.Index(self.df_dict[seed][self.columns])
            for s in in_sets_l:
                exclusive_intersection = exclusive_intersection.intersection(pd.Index(self.df_dict[s][self.columns]))
            for s in out_sets:
                exclusive_intersection = exclusive_intersection.difference(pd.Index(self.df_dict[s][self.columns]))
            final_df = self.df_dict[seed].set_index(pd.Index(self.df_dict[seed][self.columns])).ix[
                exclusive_intersection].reset_index(drop=True)
            inters_dict[in_sets] = final_df

        return in_sets_list, inters_degrees, out_sets_list, inters_dict

    def get_filtered_intersections(self, sort_by, inters_size_bounds, inters_degree_bounds):
        """

        :param columns: dict of columns, one k-v pair per dataframe.
        :return:
        """

        inters_sizes = np.array([self.inters_df_dict[x].shape[0] for x in self.in_sets_list])
        inters_degrees = np.array(self.inters_degrees)

        size_clip = (inters_sizes <= inters_size_bounds[1]) & (inters_sizes >= inters_size_bounds[0]) & (
            inters_degrees >= inters_degree_bounds[0]) & (inters_degrees <= inters_degree_bounds[1])

        in_sets_list = np.array(self.in_sets_list)[size_clip]
        out_sets_list = np.array(self.out_sets_list)[size_clip]
        inters_sizes = inters_sizes[size_clip]
        inters_degrees = inters_degrees[size_clip]

        # sort as requested
        if sort_by == 'size':
            order = np.argsort(inters_sizes)[::-1]
        elif sort_by == 'degree':
            order = np.argsort(inters_degrees)

        # store ordered data
        self.filtered_inters_sizes = inters_sizes[order]
        self.filtered_in_sets = in_sets_list[order]
        self.filtered_out_sets = out_sets_list[order]

        return self.filtered_inters_sizes, self.filtered_in_sets, self.filtered_out_sets

        # TODO: adjust figure size depending on number of graphs
        # TODO: adjust bracket size in base-set plots
        # TODO: support for: histograms, bar charts, time series - CAN THIS BE MADE COMPLETELY CUSTOM?

    def __extract_data_for_scatter(self, *, x=None, y=None):

        data_values = [dict(zip(['x', 'y', 'in_sets'],
                                [self.inters_df_dict[a][x].values,
                                 self.inters_df_dict[a][y].values,
                                 a])) for a in self.filtered_in_sets]
        return data_values


if __name__ == '__main__':
    from pickle import load

    f = open('./test_data_dict', 'rb')
    data_dict = load(f)
    f.close()
    plot(data_dict, ['title', 'rating_avg', 'rating_std'],
         additional_plots=[{'kind': 'scatter', 'data': {'x': 'rating_avg', 'y': 'rating_std'}}])