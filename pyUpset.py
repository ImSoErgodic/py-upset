__author__ = 'leo@opensignal.com'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations
import pandas as pd


def plot(data_dict, columns, sort_by='size', inters_size_bounds=(0, np.inf), inters_degree_bounds=(1, np.inf),
         additional_plots=None):
    """
    Wrapper function that initialises an UpSet class and calls its plot() method with the passed parameters.

    :param sets: set object containing the data to intersect.
    :param set_names: list-like. Must contain non-empty strings.
    :param sort_by: str. 'size | degree'.
    :param inters_size_bounds: tuple. The minimum and maximum (inclusive) size allowed for an intersection to be
    plotted.
    :param inters_degree_bounds: tuple. The minimum and maximum (inclusive) degree allowed for an intersection to
    be plotted.
    :return: figure and axes containing the plots.
    """
    return UpSet(__PlotData(data_dict)).plot(columns, sort_by, inters_size_bounds=inters_size_bounds,
                                             inters_degree_bounds=inters_degree_bounds,
                                             additional_plots=additional_plots)


class UpSet():
    def __init__(self, plot_data):
        """
        Class linked to a data set; it contains the methods to produce plots according to the UpSet representation.
        :type plot_data: __PlotData
        :param plot_data: PlotData object
        """
        self.plot_data = plot_data
        self.greys = plt.cm.Greys([.22, .8])
        self._plot_method = {
            'scatter':self._scatter
        }

    def _base_sets_plot(self, ax, sorted_sets, sorted_set_names):
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
                                        connectionstyle='arc,angleA=-180, angleB=180, armB=30' #widthB to control
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

    def _inters_sizes_plot(self, ax, inters_sizes):
        width = .5

        self._strip_axes(ax, keep_spines=['left'], keep_ticklabels=['left'])

        bar_bottom_left = self.x_values - width / 2

        ax.bar(bar_bottom_left, inters_sizes, width=width, color=self.greys[1])

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

    def _inters_matrix(self, ax, ordered_in_sets, ordered_out_sets, xlims, ylims, set_row_map):
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
            ax.scatter(np.repeat(self.x_values[col_num], len(in_y)), in_y, color=self.greys[1], s=300)
            ax.scatter(np.repeat(self.x_values[col_num], len(out_y)), out_y, color=self.greys[0], s=300)
            ax.vlines(self.x_values[col_num], min(in_y), max(in_y), lw=3.5, color=self.greys[1])

    def _create_coordinates(self, sets, inters_sizes):
        self.rows = len(sets)
        self.cols = len(inters_sizes)
        self.x_values = (np.arange(self.cols) + 1)
        self.y_values = (np.arange(self.rows) + 1)

    def _prepare_figure(self, additional_plots):

        fig = plt.figure(figsize=(16, 10))
        if additional_plots:
            main_gs = gridspec.GridSpec(3, 1, hspace=1)
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
            bot_rows, bot_cols = int(np.ceil(num_plots / 2)), 2
            gs_bottom = gridspec.GridSpecFromSubplotSpec(bot_rows, bot_cols,
                                                         subplot_spec=botgs, wspace=.3, hspace=.3)
            print(bot_rows)
            for i in range(num_plots):
                new_plotL = plt.subplot(gs_bottom[i, i%2])
                add_ax.append(new_plotL)

        return fig, (ax_intbars, ax_intmatrix, ax_setsize), tuple(add_ax)

    def plot(self, columns, sort_by='size', inters_size_bounds=(0, np.inf), inters_degree_bounds=(1, np.inf),
             additional_plots=None):
        """
        Plots intersections ignoring those with degree or size outside the boundaries passed as arguments.
        Intersections can be sorted by size or degree.
        :param columns: dict. {'name':'name of column to use in intersection'}
        :param sort_by: str. "size | degree".
        :param inters_size_bounds: tuple. The minimum and maximum (inclusive) size allowed for an intersection to be
        plotted.
        :param inters_degree_bounds: tuple. The minimum and maximum (inclusive) degree allowed for an intersection to
        be plotted.
        :return: figure and list of axes produced.
        """

        ordered_base_sets, ordered_base_set_names = self.plot_data.extract_base_sets_data(columns)
        ordered_inters_sizes, ordered_in_sets, ordered_out_sets = self.plot_data \
            .extract_intersections_data(sort_by, inters_size_bounds, inters_degree_bounds)

        self._create_coordinates(ordered_base_sets, ordered_inters_sizes)
        add_plots = ()
        fig, (ax_intbars, ax_intmatrix, ax_setsize), add_plots = self._prepare_figure(additional_plots)
        print(add_plots)

        set_row_map = dict(zip(ordered_base_set_names, self.y_values))
        ylim = self._base_sets_plot(ax_setsize, ordered_base_sets, ordered_base_set_names)
        xlim = self._inters_sizes_plot(ax_intbars, ordered_inters_sizes)
        self._inters_matrix(ax_intmatrix, ordered_in_sets, ordered_out_sets, xlim, ylim, set_row_map)

        for i, ax in enumerate(add_plots):
            print('additional')
            plot_kind = additional_plots[i]['kind']
            plot_kwargs = additional_plots[i]['kwargs']
            pm = self._plot_method[plot_kind]
            pm(ax, **(self.plot_data.extract_for_plot[plot_kind](**plot_kwargs)))


        return fig, [ax_intbars, ax_intmatrix, ax_intbars]

    def _scatter(self, ax, x_vars, y_vars, colors):
        # xvars and yvars are lists of xy couples to plot, one for each query
        for i, (x, y) in enumerate(zip(x_vars, y_vars)):
            ax.scatter(x, y, alpha=.3, color=colors[i], edgecolor=None)

        print('plotted!')

        self._strip_axes(ax, keep_spines=['bottom', 'left'], keep_ticklabels=['bottom', 'left'])

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 4))

        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        gap_y, gap_x = max(ylim) / 500.0 * 20, max(xlim) / 500.0 * 20
        ax.set_ylim(ylim[0] - gap_y, ylim[1] + gap_y)
        ax.set_xlim(xlim[0] - gap_x, xlim[1] + gap_x)
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])
        ax.spines['bottom'].set_bounds(xlim[0], ylim[0])


class __PlotData:
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.ordered_base_set_names, self.ordered_base_sets, self.set_dict = np.repeat(None, 3)
        self.extract_for_plot = {
            'scatter':self.extract_scatter_data
        }

    def extract_base_sets_data(self, columns):
        # Extract the columns as sets to intersect
        sets = []
        set_names = []
        if len(columns) != len(self.data_dict):
            raise ValueError("columns argument must specify one column per DataFrame")
        for (dfname, setname) in columns.items():
            sets.append(set(self.data_dict[dfname][setname].values))
            set_names.append('%s[%s]' % (dfname, setname))

        base_sets = np.array(sets)
        base_set_names = np.array(set_names)

        # order base sets and store
        base_sets_order = np.argsort([len(x) for x in sets])[::-1]
        self.ordered_base_sets = base_sets[base_sets_order]
        self.ordered_base_set_names = base_set_names[base_sets_order]
        self.set_dict = dict(zip(self.ordered_base_set_names, self.ordered_base_sets))

        return self.ordered_base_sets, self.ordered_base_set_names

    def extract_intersections_data(self, sort_by, inters_size_bounds, inters_degree_bounds):
        """

        :param columns: dict of columns, one k-v pair per dataframe.
        :return:
        """

        try:
            self.ordered_base_sets, self.ordered_base_set_names
        except AttributeError as e:
            print("Base sets must be extracted before it's possible to compute their intersections.")
            raise

        # Compute the intersection values
        in_sets_list = []
        out_sets_list = []
        inters_sizes = []
        inters_degrees = []

        for col_num, in_sets in enumerate(chain.from_iterable(
                combinations(self.ordered_base_set_names, i) for i in np.arange(1, len(self.ordered_base_sets) + 1))):

            inters_degrees.append(len(in_sets))
            in_sets_list.append(in_sets)
            in_sets = list(in_sets)
            out_sets = set(self.ordered_base_set_names).difference(in_sets)
            out_sets_list.append(tuple(out_sets))
            exclusive_intersection = set(self.set_dict[in_sets.pop()])

            for s in in_sets:
                exclusive_intersection.intersection_update(self.set_dict[s])
            for s in out_sets:
                exclusive_intersection.difference_update(self.set_dict[s])

            inters_sizes.append(len(exclusive_intersection))

        inters_sizes = np.array(inters_sizes)
        inters_degrees = np.array(inters_degrees)

        size_clip = (inters_sizes <= inters_size_bounds[1]) & (inters_sizes >= inters_size_bounds[0]) & (
            inters_degrees >= inters_degree_bounds[0]) & (inters_degrees <= inters_degree_bounds[1])

        in_sets_list = np.array(in_sets_list)[size_clip]
        out_sets_list = np.array(out_sets_list)[size_clip]
        inters_sizes = inters_sizes[size_clip]
        inters_degrees = inters_degrees[size_clip]

        # sort as requested
        if sort_by == 'size':
            order = np.argsort(inters_sizes)[::-1]
        elif sort_by == 'degree':
            order = np.argsort(inters_degrees)

        # store ordered data
        self.ordered_inters_sizes = inters_sizes[order]
        self.ordered_in_sets = in_sets_list[order]
        self.ordered_out_sets = out_sets_list[order]

        return self.ordered_inters_sizes, self.ordered_in_sets, self.ordered_out_sets

    def extract_scatter_data(self, x_var, y_var, query=None):
        x_vals = [df[x_var] for df in self.data_dict.values()]
        y_vals = [df[y_var] for df in self.data_dict.values()]
        colors = np.repeat('k', len(self.data_dict))
        return {'x_vars':x_vals, 'y_vars':y_vals, 'colors':colors}