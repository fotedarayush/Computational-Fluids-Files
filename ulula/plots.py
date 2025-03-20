###################################################################################################
#
# Ulula -- plots.py
#
# Plotting routines
#
# by Benedikt Diemer
#
###################################################################################################

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import ulula.utils as utils

###################################################################################################

fields = {}
"""
List of fields that can be plotted. Most fields occur in the primitive or conserved variable 
arrays, but some fields are derived (e.g., total velocity).
"""

fields["DN"] = {"name": "Density", "cmap": "viridis"}
fields["PR"] = {"name": "Pressure", "cmap": "viridis"}
fields["ET"] = {"name": "Energy", "cmap": "viridis"}
fields["VX"] = {"name": "X-velocity", "cmap": "RdBu_r"}
fields["VY"] = {"name": "Y-velocity", "cmap": "RdBu_r"}
fields["MX"] = {"name": "X-momentum", "cmap": "RdBu_r"}
fields["MY"] = {"name": "Y-momentum", "cmap": "RdBu_r"}
fields["VT"] = {"name": "Total velocity", "cmap": "viridis"}

###################################################################################################


def getPlotQuantities(sim, q_plot):
    """
    Compile an array of fluid properties

    Fluid properties are stored in separate arrays as primitive and conserved variables, or
    even in other arrays. Some quantities, such as total velocity, need to be calculated after
    the simulation has finished. This function takes care of all related operations and returns
    a single array that has the same dimensions as the domain.

    Parameters
    ----------------------------------------------------------------------------------------------
    sim: Simulation
            Object of type :data:`~ulula.simulation.Simulation`
    q_plot: array_like
            List of quantities to plot. Quantities are identified via the short strings given in the
            :data:`~ulula.plots.fields` dictionary.

    Returns
    ----------------------------------------------------------------------------------------------
    q_array: array_like
            Array of fluid properties
    """

    # Check that quantities are valid
    for q in q_plot:
        if not q in fields:
            raise Exception(
                "Unknown quantity, %s. Valid quantities are %s."
                % (str(q), str(list(fields.keys())))
            )

    nq = len(q_plot)
    q_array = np.zeros(
        (nq, sim.nx + 2 * sim.nghost, sim.ny + 2 * sim.nghost), np.float64
    )

    for iq in range(nq):
        q = q_plot[iq]
        if q in sim.q_prim:
            q_array[iq] = sim.V[sim.q_prim[q]]
        elif q in sim.q_cons:
            q_array[iq] = sim.U[sim.q_cons[q]]
        elif q == "VT":
            q_array[iq] = np.sqrt(
                sim.V[sim.q_prim["VX"]] ** 2 + sim.V[sim.q_prim["VY"]] ** 2
            )
        else:
            raise Exception("Unknown quantity, %s." % (str(q)))

    return q_array


###################################################################################################


def plot1d(
    sim,
    q_plot=["DN", "VX", "VY", "PR"],
    plot_type="line",
    idir_func=None,
    true_solution_func=None,
    vminmax_func=None,
    radial_bins_per_cell=4.0,
):
    """
    Plot fluid state along a 1D line

    Create a multi-panel plot of the fluid variables along a line through the domain. This
    plotting routine is intended for pseudo-1D simulations, where the fluid state is uniform
    in the second dimension. The line is taken at the center of the domain in that dimension.
    The plot is created but not shown or saved to a file; these operations can be completed
    using the current matplotlib figure.

    Parameters
    ----------------------------------------------------------------------------------------------
    q_plot: array_like
            List of quantities to plot. Quantities are identified via the short strings given in the
            :data:`~ulula.plots.fields` dictionary.
    plot_type: str
            The type of cut through the domain that is plotted. Can be ``line`` (in which case the
            ``idir`` parameter specifies the dimension along which the plot is made), or ``radius``
            (which creates a radially averaged plot from the center).
    idir_func: function
            If ``plot_type == line``, this function must be given and return the direction along
            which to plot (0 = x, 1 = y)
    true_solution_func: function
            If not ``None``, the given function must return a 2D array with the true solution for
            the default fluid quantities and for a given input array of coordinates. This function
            is typically implemented within the problem setup (see :doc:`setups`).
    vminmax_func: function
            A function that returns two lists of minimum and maximum plot extents for the nq
            fluid variables. If ``None``, the limits are chosen automatically.
    radial_bins_per_cell: float
            If ``plot_type == radius``, this parameter chooses how many radial bins per cell are
            plotted. The bins are averaged onto the radial annuli, so this number can be greater
            than one.
    """

    nq_plot = len(q_plot)
    q_array = getPlotQuantities(sim, q_plot)

    if plot_type == "line":

        xlabel = "x"
        idir = idir_func()
        if idir == 0:
            lo = sim.xlo
            hi = sim.xhi
            slc1d = slice(lo, hi + 1)
            slc2d = (slc1d, sim.ny // 2)
            x_plot = sim.x[slc1d]
        elif idir == 1:
            lo = sim.ylo
            hi = sim.yhi
            slc1d = slice(lo, hi + 1)
            slc2d = (sim.nx // 2, slc1d)
            x_plot = sim.y[slc1d]
        else:
            raise Exception("Unknown direction")
        xmin = x_plot[0]
        xmax = x_plot[-1]
        V_line = q_array[(slice(None),) + slc2d]

    elif plot_type == "radius":

        if (sim.nx % 2 != 0) or (sim.ny % 2 != 0):
            raise Exception(
                "For plot type radius, both nx and ny must be multiples of two (found %d, %d)"
                % (sim.nx, sim.ny)
            )
        xlabel = "r"
        slc1d = slice(None)

        # The smaller side of the domain limits the radius to which we can plot
        xmin = 0.0
        nx_half = sim.nx // 2 + sim.nghost
        ny_half = sim.ny // 2 + sim.nghost
        x_half = sim.xmax * 0.5
        y_half = sim.ymax * 0.5
        if sim.nx >= sim.ny:
            xmax = 0.5 * (sim.ymax - sim.ymin)
            n_cells = sim.nx
        else:
            xmax = 0.5 * (sim.xmax - sim.xmin)
            n_cells = sim.ny
        n_cells_half = n_cells // 2

        # Radial bins
        n_r = int(n_cells_half * radial_bins_per_cell)
        bin_edges = np.linspace(0.0, xmax, n_r + 1)
        x_plot = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Compute weight in concentric circles
        slc_x = slice(nx_half - n_cells_half, nx_half + n_cells_half)
        slc_y = slice(ny_half - n_cells_half, ny_half + n_cells_half)
        cell_x, cell_y = sim.xyGrid()
        cell_x = cell_x[(slc_x, slc_y)]
        cell_y = cell_y[(slc_x, slc_y)]
        circle_weight = np.zeros((n_r, n_cells, n_cells), np.float64)
        for i in range(n_r):
            circle_weight[i] = utils.circleSquareOverlap(
                x_half, y_half, bin_edges[i + 1], cell_x, cell_y, sim.dx
            )

        # Compute weight in bin annuli and normalize them
        bin_weight = np.zeros((n_r, n_cells, n_cells), np.float64)
        bin_weight[0] = circle_weight[0]
        for i in range(n_r - 1):
            bin_weight[i + 1] = circle_weight[i + 1] - circle_weight[i]
        bin_norm = np.sum(bin_weight, axis=(1, 2))

        # Create a square map that we use to measure the profile, then apply bin mask and sum
        V_2d = q_array[(slice(None), slc_x, slc_y)]
        V_line = (
            np.sum(bin_weight[None, :, :, :] * V_2d[:, None, :, :], axis=(2, 3))
            / bin_norm[None, :]
        )

    else:
        raise Exception("Unknown plot type, %s." % (plot_type))

    # Get true solution and min/max
    V_true = None
    if true_solution_func is not None:
        V_true = true_solution_func(sim, x_plot, q_plot)
        if (V_true is not None) and (V_true.shape[0] != nq_plot):
            raise Exception(
                "Found %d quantities in true solution, expected %d (%s)."
                % (V_true.shape[0], nq_plot, str(q_plot))
            )

    ymin = None
    ymax = None
    if vminmax_func is not None:
        ymin, ymax = vminmax_func(q_plot)
        if (ymin is not None) and (len(ymin) != nq_plot):
            raise Exception(
                "Found %d fields in lower limits, expected %d (%s)."
                % (len(ymin), nq_plot, str(q_plot))
            )
        if (ymax is not None) and (len(ymax) != nq_plot):
            raise Exception(
                "Found %d fields in upper limits, expected %d (%s)."
                % (len(ymax), nq_plot, str(q_plot))
            )

    # Prepare figure
    panel_size = 3.0
    space = 0.15
    space_lb = 1.0
    fwidth = space_lb + panel_size * nq_plot + space_lb * (nq_plot - 1) + space
    fheight = space_lb + panel_size + space
    fig = plt.figure(figsize=(fwidth, fheight))
    gs = gridspec.GridSpec(1, nq_plot)
    plt.subplots_adjust(
        left=space_lb / fwidth,
        right=1.0 - space / fwidth,
        bottom=space_lb / fheight,
        top=1.0 - space / fheight,
        hspace=space_lb / panel_size,
        wspace=space_lb / panel_size,
    )

    # Create panels
    panels = []
    for i in range(nq_plot):
        panels.append(fig.add_subplot(gs[i]))
        plt.xlim(xmin, xmax)
        if (ymin is not None) and (ymin[i] is not None) and (ymax[i] is not None):
            plt.ylim(ymin[i], ymax[i])
        plt.xlabel(xlabel)
        plt.ylabel(fields[q_plot[i]]["name"])

    # Plot fluid variables
    for i in range(nq_plot):
        plt.sca(panels[i])
        if V_true is not None:
            plt.plot(
                x_plot, V_true[i, :], ls="-", color="deepskyblue", label="True solution"
            )
        plt.plot(
            x_plot, V_line[i, :], color="darkblue", label="Solution, t=%.2f" % (sim.t)
        )

    # Finalize plot
    plt.sca(panels[0])
    plt.legend(loc=1, labelspacing=0.05)

    return


###################################################################################################


def plot2d(
    sim,
    q_plot=["DN", "VX", "VY", "PR"],
    vminmax_func=None,
    cmap_func=None,
    panel_size=3.0,
    plot_ghost_cells=False,
):
    """
    Plot fluid state in 2D

    Create a multi-panel plot of the fluid variables along a line through the domain. This
    plotting routine is intended for pseudo-1D simulations, where the fluid state is uniform
    in the second dimension. The line is taken at the center of the domain in that dimension.
    The plot is created but not shown or saved to a file; these operations can be completed
    using the current matplotlib figure.

    Parameters
    ----------------------------------------------------------------------------------------------
    q_plot: array_like
            List of quantities to plot. Quantities are identified via the short strings given in the
            :data:`~ulula.plots.fields` dictionary.
    vminmax_func: function
            A function that returns two lists of minimum and maximum plot extents for the nq
            fluid variables. If ``None``, the limits are chosen automatically.
    cmap_func: function
            A function that returns a list of size nq with colormap objects to be used when
            plotting the fluid variables. If ``None``, the default colormap is used for all
            fluid variables.
    panel_size: float
            Size of each plotted panel in inches
    plot_ghost_cells: bool
            If ``True``, ghost cells are plotted and separated from the physical domain by a gray
            frame. This option is useful for debugging.
    """

    # Constants
    space = 0.15
    space_lb = 0.8
    cbar_width = 0.2

    # Compute quantities
    nq_plot = len(q_plot)
    q_array = getPlotQuantities(sim, q_plot)

    # Get x-extent
    if plot_ghost_cells:
        xlo = 0
        xhi = sim.nx + 2 * sim.nghost - 1
        ylo = 0
        yhi = sim.ny + 2 * sim.nghost - 1

        xmin = sim.x[0] - 0.5 * sim.dx
        xmax = sim.x[-1] + 0.5 * sim.dx
        ymin = sim.y[0] - 0.5 * sim.dx
        ymax = sim.y[-1] + 0.5 * sim.dx
    else:
        xlo = sim.xlo
        xhi = sim.xhi
        ylo = sim.ylo
        yhi = sim.yhi

        xmin = sim.xmin
        xmax = sim.xmax
        ymin = sim.ymin
        ymax = sim.ymax

    slc_x = slice(xlo, xhi + 1)
    slc_y = slice(ylo, yhi + 1)
    xext = xmax - xmin
    yext = ymax - ymin

    # Prepare figure; take the larger dimension and assign that the panel size; the smaller
    # dimension follows from that.
    if xext >= yext:
        panel_w = panel_size
        panel_h = yext / xext * panel_w
    else:
        panel_h = panel_size
        panel_w = xext / yext * panel_h

    fwidth = space_lb + (panel_w + space) * nq_plot
    fheight = space_lb + panel_h + space + cbar_width + space_lb

    fig = plt.figure(figsize=(fwidth, fheight))
    gs = gridspec.GridSpec(
        3, nq_plot, height_ratios=[space_lb * 0.8, cbar_width, panel_h]
    )
    plt.subplots_adjust(
        left=space_lb / fwidth,
        right=1.0 - space / fwidth,
        bottom=space_lb / fheight,
        top=1.0 - space / fheight,
        hspace=space / fheight,
        wspace=space / panel_w,
    )

    # Create panels
    panels = []
    for i in range(nq_plot):
        panels.append([])
        for j in range(3):
            panels[i].append(fig.add_subplot(gs[j, i]))

            if j == 0:
                plt.axis("off")
            elif j == 1:
                pass
            else:
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.xlabel("x")
                if i == 0:
                    plt.ylabel("y")
                else:
                    plt.gca().set_yticklabels([])

    # Check for plot limits and colormaps specific to the setup
    vmin = None
    vmax = None
    if vminmax_func is not None:
        vmin, vmax = vminmax_func(q_plot)
        if (vmin is not None) and (len(vmin) != nq_plot):
            raise Exception(
                "Found %d fields in lower limits, expected %d (%s)."
                % (len(vmin), nq_plot, str(q_plot))
            )
        if (vmax is not None) and (len(vmax) != nq_plot):
            raise Exception(
                "Found %d fields in upper limits, expected %d (%s)."
                % (len(vmax), nq_plot, str(q_plot))
            )

    cmaps = None
    if cmap_func is not None:
        cmaps = cmap_func(q_plot)
        if (cmaps is not None) and (len(cmaps) != nq_plot):
            raise Exception(
                "Found %d fields in colormaps, expected %d (%s)."
                % (len(cmaps), nq_plot, str(q_plot))
            )

    # Plot fluid variables
    for i in range(nq_plot):
        plt.sca(panels[i][2])
        data = q_array[i, slc_x, slc_y]
        data = data.T[::-1, :]

        if (vmin is None) or (vmin[i] is None):
            vmin_ = np.min(data)
        else:
            vmin_ = vmin[i]
        if (vmax is None) or (vmax[i] is None):
            vmax_ = np.max(data)
        else:
            vmax_ = vmax[i]

        if (cmaps is None) or (cmaps[i] is None):
            cmap = plt.get_cmap(fields[q_plot[i]]["cmap"])
        else:
            cmap = cmaps[i]

        norm = mpl.colors.Normalize(vmin=vmin_, vmax=vmax_)

        plt.imshow(
            data,
            extent=[xmin, xmax, ymin, ymax],
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )

        ax = panels[i][1]
        plt.sca(ax)
        cb = mpl.colorbar.ColorbarBase(
            ax, orientation="horizontal", cmap=cmap, norm=norm
        )
        cb.set_label(fields[q_plot[i]]["name"], rotation=0, labelpad=8)
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        cb.ax.xaxis.set_tick_params(pad=5)

        # Plot frame around domain if plotting ghost cells
        if plot_ghost_cells:
            plt.sca(panels[i][2])
            plt.plot(
                [sim.xmin, sim.xmax, sim.xmax, sim.xmin, sim.xmin],
                [sim.ymin, sim.ymin, sim.ymax, sim.ymax, sim.ymin],
                "-",
                color="gray",
            )

    return


###################################################################################################
