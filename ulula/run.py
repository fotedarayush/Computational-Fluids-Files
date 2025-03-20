###################################################################################################
#
# Ulula -- run.py
#
# Runtime for the Ulula simultion code. The run function is a simple wrapper around the simulation
# object that runs a given setup and creates snapshot files, plots, and movies.
#
# by Benedikt Diemer
# Slightly adapted by Paul McMillan (2025)
#
###################################################################################################

import os
import subprocess
import glob
from matplotlib import pyplot as plt
import copy
import time
import math

import ulula.simulation as ulula_sim
import ulula.plots as ulula_plots

###################################################################################################


def run(
    setup,
    hydro_scheme=None,
    nx=200,
    tmax=1.0,
    max_steps=None,
    print_step=100,
    restart_file=None,
    output_step=None,
    output_time=None,
    output_suffix="",
    plot_step=None,
    plot_time=None,
    plot_ics=True,
    plot1d=False,
    save_plots=True,
    plot_dir=".",
    plot_suffix="",
    plot_file_ext="png",
    plot_dpi=300,
    movie=False,
    movie_length=4.0,
    movie_fps=25,
    movie_dpi=200,
    **kwargs
):
    """
    Runtime environment for Ulula.

    This function takes a given problem setup and other user-defined parameters and executes the
    hydro solver. Depending on user choices, it can also produces output files, plots, and movies.
    Customizations that are implemented in the setup class (e.g., which variables to plot with
    which colormaps) are automatically routed to the respective plotting routines.

    Parameters
    -----------------------------------------------------------------------------------------------
    setup: Setup
            Setup object. See :doc:`setups` for how to create this object.
    hydro_scheme: HydroScheme
            HydroScheme object that sets the algorithm and CFL number for the simulation. If ``None``,
            the standard scheme is used. See :doc:`simulation` for details.
    nx: int
            Number of cells in the x-direction. The ratio of x and y is determined by the problem
            setup.
    tmax: float
            Time when the simulation should be stopped (in code units).
    max_steps: int
            Maximum number of steps to take. If ``None``, no limit is imposed and the code is run to
            a time ``tmax``.
    print_step: int
            Print a line to the console every ``print_step`` timesteps.
    restart_file: str
            If not ``None``, the simulation is loaded from this filename and restarted at the step
            where it was saved. The setup is ignored.
    output_step: int
            Output a snapshot/restart file every ``output_step`` timesteps. Note that this spacing
            probably does not correspond to fixed times. If the latter is desired, use
            ``output_time``. Both ``output_step`` and ``output_time`` can be used at the same time
            to produce two sets of files.
    output_time: float
            Produce output files in time intervals of size ``output_time`` (given in code units). This
            parameter should not change the progression of the simulation because the timesteps taken
            to arrive at the desired times are not used for the actual simulation.
    output_suffix: string
            String to add to all output filenames.
    plot_step: int
            Produce a plot every ``plot_step`` timesteps. Note that this spacing probably does not
            correspond to fixed times. If the latter is desired, use ``plot_time``. Both ``plot_step``
            and ``plot_time`` can be used at the same time to produce two sets of plots.
    plot_time: float
            Produce plots in time intervals of size ``plot_time`` (given in code units). This
            parameter should not change the progression of the simulation because the timesteps taken
            to arrive at the desired times are not used for the actual simulation.
    plot_ics: bool
            Produce a plot of the initial conditions, step 0 (only active if ``plot_step == True``)
    plot1d: bool
            If ``True``, the 1D plotting routine is called instead of the usual 2D routine. This is
            useful only for test setups that are intrinsically 1D such as a shocktube.
    save_plots: bool
            If ``True``, plots are saved to a file (see also ``plot_suffix``, ``plot_file_ext``,
            and ``plot_dpi``). If ``False``, plots are shown in an interactive matplotlib window. Note
            that this can happen many times during a simulation depending on ``plot_step`` and/or
            ``plot_time``.
    plot_dir: string
            Directory where plots are saved. If default, the current directory is used.
    plot_suffix: string
            String to add to all plot filenames (only active if ``save_plots == True``)
    plot_file_ext: string
            File extension for plots; can be ``png``, ``pdf``, or any other extension supported by
            the matplotlib library (only active if ``save_plots == True``).
    plot_dpi
            Dots per inch for png figures (only active if ``save_plots == True`` and
            ``plot_file_ext == png`` or other bitmap-like image formats).
    movie: bool
            If ``True``, a movie is created by outputting a frame at equally spaced times and running
            the ffmpeg tool to combine them (this tool must be installed on the system). See also
            ``movie_length``, ``movie_fps``, and ``movie_dpi``.
    movie_length: float
            Length of the movie in seconds (not code units!)
    movie_fps: int
            Framerate of the movie (25 is typical)
    movie_dpi: int
            Resolution of the png files used to create the movie (see ``plot_dpi``)
    kwargs: kwargs
            Additional arguments that are passed to the Ulula plotting function (either 1D or 2D,
            depending on the ``plot1d`` parameter).

    Returns
    -----------------------------------------------------------------------------------------------
    sim: Simulation
            Object of type :data:`~ulula.simulation.Simulation`
    """

    print("running...")
    next_time_output = None
    next_time_plot = None
    next_time_movie = None

    if movie:
        step_movie = 0
        movie_time = tmax / (movie_length * movie_fps - 1)
    else:
        step_movie = None
        movie_time = None

    setup_name = setup.shortName()

    # Plotting settings. For simplicity, we only allow either plotting in 1D or 2D so that any
    # keyword arguments can be routed to the plotting function. If both were allowed, we would need
    # to distinguish which keyword arguments are meant for which of the two functions.
    plot_kwargs = copy.copy(kwargs)
    if plot1d:
        plotFunction = ulula_plots.plot1d
        plot_kwargs.update(
            dict(
                idir_func=setup.direction,
                true_solution_func=setup.trueSolution,
                vminmax_func=setup.plotLimits,
            )
        )
        plot_suffix = "%s_1d" % (plot_suffix)
    else:
        plotFunction = ulula_plots.plot2d
        plot_kwargs.update(
            dict(vminmax_func=setup.plotLimits, cmap_func=setup.plotColorMaps)
        )
    # Check that the plot directory exists, if not create it
    if save_plots:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

    # ---------------------------------------------------------------------------------------------

    # Perform step-based saving and plotting operations

    def checkOutputStep(sim, final_step=False):

        if (output_step is not None) and (sim.step % output_step == 0):
            sim.save(filename="ulula_step_%04d%s.hdf5" % (sim.step, output_suffix))

        if (
            (plot_step is not None)
            and ((sim.step % plot_step == 0) or final_step)
            and not ((sim.step == 0) and (plot_ics == False))
        ):
            plotFunction(sim, **plot_kwargs)
            if save_plots:
                print(plot_dir)
                plt.savefig(
                    plot_dir
                    + "/ulula_%s_step_%04d%s.%s"
                    % (setup_name, sim.step, plot_suffix, plot_file_ext)
                )
                plt.close()
            else:
                plt.show()

        return

    # Compute the next time when a certain operation needs to happen given the current time and the
    # operation's time interval.

    def nextTime(sim, interval):

        if interval is None:
            next_time = None
        else:
            n_t = math.floor(sim.t / interval)
            next_time = interval * (n_t + 1)

        return next_time

    # If a particular operation needs to happen at t_next and that time is within the next
    # timestep, we need to return the simulation at time t_next. To avoid messing with the actual
    # simulation run by inserting an artificially small timestep, we copy the entire simulation
    # object and advance it by the desired timestep. This operation has some memory overhead but
    # is cleaner than trying to restore the previous state to the main simulation object.

    def getSimAtTime(sim, dt_next, t_next):

        if t_next is None:
            return False, None

        do_operation = sim.t + dt_next >= t_next

        if do_operation:
            dt_needed = t_next - sim.t
            if abs(dt_needed) < 1e-7 * tmax:
                sim_copy = sim
            else:
                sim_copy = copy.copy(sim)
                sim_copy.timestep(dt=dt_needed)
        else:
            sim_copy = None

        return do_operation, sim_copy

    # ---------------------------------------------------------------------------------------------

    # If a restart file is given, we load it and start the simulation from the respective snapshot.
    if restart_file is not None:
        sim = ulula_sim.load(restart_file)
        if sim.t >= tmax:
            raise Exception(
                "The final time tmax (%.2e) must be greater than the time in the restart file (%.2e)."
                % (tmax, sim.t)
            )
        next_time_output = nextTime(sim, output_time)
        next_time_plot = nextTime(sim, plot_time)
        next_time_movie = nextTime(sim, movie_time)

    else:
        # Create simulation object and set initial conditions
        if hydro_scheme is None:
            hydro_scheme = ulula_sim.HydroScheme()
        sim = ulula_sim.Simulation(hydro_scheme)
        setup.initialConditions(sim, nx)

        # Plot/save initial conditions and reset the next operation times to 0 so that time-based
        # saving/plotting is also performed at t = 0.
        checkOutputStep(sim)
        if output_time is not None:
            next_time_output = 0.0
        if plot_time is not None:
            next_time_plot = 0.0
        if movie_time is not None:
            next_time_movie = 0.0

    # Main loop over timesteps. We record the starting timestep as it may not be zero if we
    # are restarting from a file.
    t0 = time.process_time()
    step_start = sim.step

    while sim.t < tmax:

        # Compute timestep. Before we actually do the timestep, we need to check whether we need to
        # output or plot the simulation at a particular time during this timestep.
        dt = sim.cflCondition()

        # Check whether we need to output a snapshot file during the next timestep
        do_output, sim_copy = getSimAtTime(sim, dt, next_time_output)
        if do_output:
            sim_copy.save(
                filename="ulula_time_%.4f%s.hdf5" % (sim_copy.t, output_suffix)
            )

        # Check whether we need to create a plot during the next timestep
        do_plot, sim_copy = getSimAtTime(sim, dt, next_time_plot)
        if do_plot:
            plotFunction(sim_copy, **plot_kwargs)
            if save_plots:
                plt.savefig(
                    plot_dir
                    + "/ulula_%s_time_%.4f%s.%s"
                    % (setup_name, sim_copy.t, plot_suffix, plot_file_ext),
                    dpi=plot_dpi,
                )
                plt.close()
            else:
                plt.show()

        # Check whether we need to output a movie frame during the next timestep
        do_movie, sim_copy = getSimAtTime(sim, dt, next_time_movie)
        if do_movie:
            plotFunction(sim_copy, **plot_kwargs)
            plt.savefig("frame_%04d.png" % (step_movie), dpi=movie_dpi)
            plt.close()
            step_movie += 1

        # Perform the actual timestep
        sim.timestep(dt=dt)
        if sim.step % print_step == 0:
            print("Timestep %5d, dt = %.2e, t = %.2e" % (sim.step, dt, sim.t))

        # Set the next times for output/plotting/movie frames
        if do_output:
            next_time_output = nextTime(sim, output_time)
        if do_plot:
            next_time_plot = nextTime(sim, plot_time)
        if do_movie:
            next_time_movie = nextTime(sim, movie_time)

        # Save and/or plot at this step if necessary
        checkOutputStep(sim)

        # Check for abort conditions
        if (max_steps is not None) and (sim.step >= max_steps):
            break

    # Print timing info
    ttot = time.process_time() - t0
    steps_taken = sim.step - step_start
    print(
        "Simulation finished. Took %d steps, %.1f seconds, %.3f s/step, %.2f steps/s."
        % (steps_taken, ttot, ttot / steps_taken, steps_taken / ttot)
    )

    # Render movie
    if movie:
        cmd_str = "ffmpeg -i frame_%04d.png -pix_fmt yuv420p -y"
        cmd_str += " -framerate %d" % (movie_fps)
        cmd_str += " ulula_%s%s.mp4" % (setup_name, plot_suffix)
        subprocess.run(cmd_str, shell=True)
        for frame in glob.glob("frame*.png"):
            try:
                os.remove(frame)
            except OSError:
                pass

    # Plot final state
    checkOutputStep(sim, final_step=True)

    return sim


###################################################################################################
