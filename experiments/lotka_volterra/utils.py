import os
import warnings

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase

import torch

from symplearn.datasets import VectorFieldDataset, SnapshotDataset

from models import LotkaVolterra, NeuralBaseLV, NeuralSympLV


def init_env():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")

    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 9

    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=BIGGER_SIZE, title_fontsize=BIGGER_SIZE)  # legend fontsize

    pb = LotkaVolterra()
    models = load_models()
    return pb, models


# https://stackoverflow.com/questions/55501860/how-to-put-multiple-colormap-patches-in-a-matplotlib-legend
class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, cmin=0.0, cmax=1.0, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.cmin, self.cmax = cmin, cmax

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        stripes = []
        w = width / self.num_stripes
        h = height
        lin_color = (0.5 + np.arange(self.num_stripes)) / self.num_stripes
        c = self.cmap(self.cmin + (self.cmax - self.cmin) * lin_color)
        for i, ci in enumerate(c):  # range(self.num_stripes):
            pos = [xdescent + i * width / self.num_stripes, ydescent]
            s = Rectangle(pos, w, h, fc=ci, transform=trans)
            # fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
            stripes.append(s)
        return stripes

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {}
    # default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    c = np.asarray(c)
    if c.ndim == 1:
        lc = LineCollection(segments, **default_kwargs)
        lc.set_array(c)  # set the colors of each segment
    else:
        lc = LineCollection(segments, colors=c, **default_kwargs)

    return ax.add_collection(lc)


def find_index_period(z, init_offset=2):
    d = np.square(z - z[0]).sum(-1)
    (idx,) = np.where(d[init_offset:] < d[init_offset])
    return idx[0] + (3 * init_offset + 1) // 2


def get_reduced_val_data():
    model = LotkaVolterra()
    val_data = VectorFieldDataset("val")

    z_all = val_data.z
    x_all, y_all = z_all.tensor_split(2, -1)
    h_all = model.hamiltonian(x_all, y_all, None)

    idx_sort = torch.argsort(h_all)
    n = len(z_all)
    idx = -(n // np.arange(2, 7))
    idx = -(np.array([75, 55, 40, 15]) * n) // 100
    z_sorted = z_all[idx_sort]
    z0 = z_sorted[idx]

    t0 = torch.zeros_like(z0)[:, 0]

    data_dvi = SnapshotDataset("val")
    dt = data_dvi.dt
    return t0, z0, dt


def get_extreme_val_data():
    model = LotkaVolterra()
    val_data = VectorFieldDataset("val")

    z_all = val_data.z
    x_all, y_all = z_all.tensor_split(2, -1)
    h_all = model.hamiltonian(x_all, y_all, None)

    idx_sort = torch.argsort(h_all)
    n = len(z_all)
    idx = [-(n // 4), -(n // 5), -10, -4, -1]
    z_sorted = z_all[idx_sort]
    z0 = z_sorted[idx]

    t0 = torch.zeros_like(z0)[:, 0]

    data_dvi = SnapshotDataset("val")
    dt = data_dvi.dt
    return t0, z0, dt


def load_models():
    models = {}
    models["ref"] = LotkaVolterra()

    models["vf_base"] = NeuralBaseLV()
    models["vf_base"].load_state_dict(
        torch.load(os.path.join("nn", "baseline.pt"), weights_only=True)
    )
    models["vf_base"].eval()

    models["vf_reg"] = NeuralSympLV()
    models["vf_reg"].load_state_dict(
        torch.load(os.path.join("nn", "vf_reg.pt"), weights_only=True)
    )
    models["vf_reg"].eval()

    models["vf_no_reg"] = NeuralSympLV()
    models["vf_no_reg"].load_state_dict(
        torch.load(os.path.join("nn", "vf_no_reg.pt"), weights_only=True)
    )
    models["vf_no_reg"].eval()

    models["dvi"] = NeuralSympLV()
    models["dvi"].load_state_dict(
        torch.load(os.path.join("nn", "dvi.pt"), weights_only=True)
    )
    models["dvi"].eval()

    return models
