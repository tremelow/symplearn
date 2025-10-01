import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase

import torch

from symplearn.datasets import SnapshotDataset, VectorFieldDataset
from symplearn.training import VectorFieldLoss
from symplearn.training.norms import GramMSNorm

from models import GuidingCenter, NeuralBaseGC, NeuralSympGC


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


def find_index_period(z, eps_dist=1e-4, eps_diff=0.0):
    d = np.sqrt(np.square(z - z[0]).sum(-1))
    diff_d = np.diff(d)
    loc_min = (-eps_diff < diff_d[1:]) & (diff_d[:-1] < eps_diff)
    (idx,) = np.where((d[1:-1] < eps_dist) & loc_min)
    return idx[0] + 1


def get_study_init_cond():
    u0 = torch.tensor(
        [
            # -8.612e-4,  # deeply passing
            -7.782e-4,  # barely passing
            -7.610e-4,  # barely trapped (ill-captured by DVI)
            -7.487e-4,  # barely trapped (well captured by DVI)
            -4.306e-4,  # deeply trapped (banana orbit)
        ]
    )
    z0 = torch.tensor([0.0, 0.0, 0.05, 0.0]).repeat(len(u0), 1)
    z0[:, 3] = u0

    t0 = torch.zeros_like(z0)[:, 0]
    dt = SnapshotDataset("val").dt

    return t0, z0, dt


def plot_rz(ax, z, colors=None, model=GuidingCenter()):
    th, _, r, _ = z.unbind(dim=-1)
    R, Z = model.r0 + r * torch.cos(th), model.r0 + r * torch.sin(th)

    kwargs = {"s": 1}
    if colors is not None:
        kwargs["c"] = colors
    return ax.scatter(R, Z, **kwargs)


def plot_thu(ax, z, colors=None):
    th, _, _, u = z.unbind(dim=-1)
    th = (th + np.pi) % (2 * np.pi) - np.pi

    kwargs = {"s": 1}
    if colors is not None:
        kwargs["c"] = colors
    return ax.scatter(th, u, **kwargs)


def plot_z(z, colors=None, model=GuidingCenter()):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    kwargs = {"s": 1}
    if colors is not None:
        kwargs["c"] = colors
    plot1 = plot_rz(ax1, z, colors=colors, model=model)
    plot_thu(ax2, z, colors=colors)
    fig.colorbar(plot1, ax=(ax1, ax2))
    return fig


def test_reg(models):
    data_vf = VectorFieldDataset("val")
    z, t, dt_z = data_vf[:]
    reg_fun = lambda model: torch.vmap(VectorFieldLoss(model)._error_reg)
    compute_reg = lambda model: reg_fun(model)(z, t, dt_z)[1]

    reg_ref = compute_reg(models["ref"])
    reg_vf_reg = compute_reg(models["vf_reg"])
    reg_vf_no_reg = compute_reg(models["vf_no_reg"])

    norm = lambda reg_val: GramMSNorm()(reg_val, dt_z)
    return norm(reg_ref), norm(reg_vf_reg), norm(reg_vf_no_reg)


def load_models():
    models = {}

    models["ref"] = GuidingCenter()

    models["vf_base"] = NeuralBaseGC()
    models["vf_base"].load_state_dict(
        torch.load(os.path.join("nn", "baseline.pt"), weights_only=True)
    )
    models["vf_base"].eval()

    models["vf_reg"] = NeuralSympGC()
    models["vf_reg"].load_state_dict(
        torch.load(os.path.join("nn", "vf_reg.pt"), weights_only=True)
    )
    models["vf_reg"].eval()

    models["vf_no_reg"] = NeuralSympGC()
    models["vf_no_reg"].load_state_dict(
        torch.load(os.path.join("nn", "vf_no_reg.pt"), weights_only=True)
    )
    models["vf_no_reg"].eval()

    models["vf_no_gram"] = NeuralSympGC()
    models["vf_no_gram"].load_state_dict(
        torch.load(os.path.join("nn", "vf_no_gram.pt"), weights_only=True)
    )
    models["vf_no_gram"].eval()

    models["dvi"] = NeuralSympGC()
    models["dvi"].load_state_dict(
        torch.load(os.path.join("nn", "dvi.pt"), weights_only=True)
    )
    models["dvi"].eval()

    return models
