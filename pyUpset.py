__author__ = 'leo@opensignal.com'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations
import pandas as pd

def plot(sets, set_names, sort_by='size', inters_size_bounds=(0, np.inf), inters_degree_bounds=(1, np.inf)):
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
    return UpSet(sets, set_names).plot(sort_by, inters_size_bounds=inters_size_bounds,
                                inters_degree_bounds=inters_degree_bounds)


class UpSet():

    def __init__(self, data_dict):
        """
        Class linked to a data set; it contains the methods to produce plots according to the UpSet representation.

        :param data_dict: dict. {'name': pandas DataFrame}
        """
        self.data_dict = data_dict
        self.inters_degree_bounds, self.inters_size_bounds = None, None

    def _base_sets_plot(self, ax, sorted_sets, sorted_set_names):
        ax.invert_xaxis()
        height = .6
        bar_bottoms = self.y_values - height / 2

        greys = plt.cm.Greys([.2, .8])

        ax.barh(bar_bottoms, [len(x) for x in sorted_sets], height=height, color=greys[1])

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 4))

        ax.tick_params(axis='both',
                       which='both',
                       bottom='on',
                       top='off',
                       left='off',
                       right='off',
                       labelbottom='on',
                       labeltop='off',
                       labelleft='off',
                       labelright='off')

        for s in ax.spines.items():
            if s[0] != 'bottom':
                s[1].set_visible(False)

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
                                        connectionstyle='arc,angleA=-180, angleB=180, armB=30'
                                        ),
                        )

        ax.set_xlabel("Set size", fontweight='bold', fontsize=13)

        return ax.get_ylim()

    def _inters_sizes_plot(self, ax, inters_sizes):
        width = .5

        greys = plt.cm.Greys([.2, .8])

        ax.tick_params(which='both',
                       bottom='off',
                       top='off',
                       left='on',
                       right='off',
                       labelbottom='off',
                       labeltop='off',
                       labelleft='on',
                       labelright='off')

        for sname, spine in ax.spines.items():
            if sname != 'left':
                spine.set_visible(False)

        bar_bottom_left = self.x_values - width / 2

        ax.bar(bar_bottom_left, inters_sizes, width=width, color=greys[1])

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

        ax.tick_params(axis='both',
                       which='both',
                       bottom='off',
                       top='off',
                       left='off',
                       right='off',
                       labelbottom='off',
                       labeltop='off',
                       labelleft='off',
                       labelright='off'
                       )

        greys = plt.cm.Greys([.22, .8])
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
            # in_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=greys[1]) for y in in_y]
            # out_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=greys[0]) for y in out_y]
            # for c in chain.from_iterable([in_circles, out_circles]):
            # ax.add_patch(c)
            ax.scatter(np.repeat(self.x_values[col_num], len(in_y)), in_y, color=greys[1], s=300)
            ax.scatter(np.repeat(self.x_values[col_num], len(out_y)), out_y, color=greys[0], s=300)
            ax.vlines(self.x_values[col_num], min(in_y), max(in_y), lw=3.5, color=greys[1])
            for s in ax.spines.values():
                s.set_visible(False)

    def _compute_intersections(self, set_dict, inters_size_bounds=(0, np.inf), inters_degree_bounds=(0, np.inf)):
        """
        Computes the intersections of the sets passed to the class discarding those beyond the size or degree bounds.

        :param set_dict: dict. {'name':'set'}
        :param inters_size_bounds: tuple. The minimum and maximum (inclusive) size allowed for an intersection to be
        plotted.
        :param inters_degree_bounds: tuple. The minimum and maximum (inclusive) degree allowed for an intersection to
        be plotted.
        :returns self.
        """
        in_sets_list = []
        out_sets_list = []
        inters_sizes = []
        inters_degrees = []

        set_names = set_dict.keys()

        for col_num, in_sets in enumerate(chain.from_iterable(
                combinations(set_names, i) for i in np.arange(1, len(set_dict) + 1))):

            inters_degrees.append(len(in_sets))

            in_sets_list.append(in_sets)
            in_sets = list(in_sets)

            out_sets = set(set_names).difference(in_sets)
            out_sets_list.append(tuple(out_sets))

            exclusive_intersection = set(set_dict[in_sets.pop()])
            for s in in_sets:
                exclusive_intersection.intersection_update(set_dict[s])
            for s in out_sets:
                exclusive_intersection.difference_update(set_dict[s])
            inters_sizes.append(len(exclusive_intersection))

        inters_sizes = np.array(inters_sizes)
        inters_degrees = np.array(inters_degrees)

        size_clip = (inters_sizes <= inters_size_bounds[1]) & (inters_sizes >= inters_size_bounds[0]) & (
            inters_degrees >= inters_degree_bounds[0]) & (inters_degrees <= inters_degree_bounds[1])

        self.in_sets_list = np.array(in_sets_list)[size_clip]
        self.out_sets_list = np.array(out_sets_list)[size_clip]
        self.inters_sizes = inters_sizes[size_clip]
        self.inters_degrees = inters_degrees[size_clip]
#        self.inters_size_bounds, self.inters_degree_bounds = inters_size_bounds, inters_degree_bounds

        return self


    def _create_coordinates(self, sets, inters_sizes):
        self.rows = len(sets)
        self.cols = len(inters_sizes)
        self.x_values = (np.arange(self.cols) + 1)
        self.y_values = (np.arange(self.rows) + 1)

    def _prepare_figure(self):
        fig = plt.figure(figsize=(16, 10))
        setsize_w, setsize_h = 3, self.rows
        intmatrix_w, intmatrix_h = setsize_w + self.cols, self.rows
        intbars_w, intbars_h = setsize_w + self.cols, self.rows * 4
        fig_cols = self.cols + 3
        fig_rows = self.rows + self.rows * 4
        gs = gridspec.GridSpec(fig_rows, fig_cols)
        gs.update(wspace=.1, hspace=.2)
        ax_setsize = plt.subplot(gs[-1:-setsize_h, 0:setsize_w])
        ax_intmatrix = plt.subplot(gs[-1:-intmatrix_h, setsize_w:intmatrix_w])
        ax_intbars = plt.subplot(gs[:self.rows * 4 - 1, setsize_w:intbars_w])
        return ax_intbars, ax_intmatrix, ax_setsize, fig

    def plot(self, columns, sort_by='size', inters_size_bounds=(0, np.inf), inters_degree_bounds=(1, np.inf)):
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

        sets = []
        set_names = []
        for (dfname, setname) in columns.items():
            sets.append(set(self.data_dict[dfname][setname].values))
            set_names.append('%s[%s]' % (dfname, setname))
        sets = np.array(sets)
        set_names = np.array(set_names)

        # if (self.inters_size_bounds != inters_size_bounds) and (self.inters_degree_bounds != inters_degree_bounds):
        self._compute_intersections(dict(zip(set_names, sets)), inters_size_bounds, inters_degree_bounds)

        if sort_by == 'size':
            order = np.argsort(self.inters_sizes)[::-1]
        elif sort_by == 'degree':
            order = np.argsort(self.inters_degrees)

        ordered_inters_sizes = self.inters_sizes[order]
        ordered_in_sets = self.in_sets_list[order]
        ordered_out_sets = self.out_sets_list[order]

        self._create_coordinates(sets, ordered_inters_sizes)

        ax_intbars, ax_intmatrix, ax_setsize, fig = self._prepare_figure()

        base_sets_order = np.argsort([len(x) for x in sets])[::-1]
        sorted_sets = sets[base_sets_order]
        sorted_sets_names = set_names[base_sets_order]
        set_row_map = dict(zip(sorted_sets_names, self.y_values))

        ylim = self._base_sets_plot(ax_setsize, sorted_sets, sorted_sets_names)
        xlim = self._inters_sizes_plot(ax_intbars, ordered_inters_sizes)
        self._inters_matrix(ax_intmatrix, ordered_in_sets, ordered_out_sets, xlim, ylim, set_row_map)

        return fig, [ax_intbars, ax_intmatrix, ax_intbars]

    def _scatter(self, ax, xvar, yvar):
        pass