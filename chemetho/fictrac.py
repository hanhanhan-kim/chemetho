#!/usr/bin/env python3

# TODO: Sym link my local into noexiit src? How to rid of path.insert?

"""
Process and visualize FicTrac data with helper functions. 
When run as a script, transforms .dat FicTrac files into a single concatenated 
Pandas dataframe with some additional columns. Then performs various processing 
and plotting of the FicTrac data. Includes individual visualizations of 
the frequency domain, low-pass Butterworth filtering, XY path with a colour map, ___. 
Includes population visualizations, such as histograms, ECDFs, and __. 
"""

# import glob
from sys import exit
from shutil import rmtree
from os.path import join, exists
from pathlib import Path
from os import mkdir
import datetime
import re

import yaml
import numpy as np
import pandas as pd

from .fourier import fft
from .common import bw_filter, flatten_list, search_for_paths, add_metadata_to_dfs, regenerate_IDs
from . import viz


def get_datetime_from_logs(log, acq_mode="online"):

    """
    Extract 't_sys' (ms) from the FicTrac .log files. 
    """

    assert (acq_mode == "online"), \
        "This function applies only to FicTrac data acquired in real-time"
    with open (log, "r") as f:

        log_lines = f.readlines()

        datetime_list = []
        for line in log_lines:
            if "Frame captured " in line:
                # Pull out substring between t_sys and ms:
                result = re.search("t_sys: (.*?) ms", line)
                # Convert from ms to s: 
                t_sys = float(result.group(1)) / 1000
                datetime_obj = datetime.datetime.fromtimestamp(t_sys)
                datetime_list.append(datetime_obj)
        
        # FicTrac logs a t_sys before frame 0. Get rid of it:
        del datetime_list[0]

    return datetime_list


def parse_dats(root, ball_radius, acq_mode, do_confirm=True):

    """
    Batch processes subdirectories, where each subdirectory is labelled 'fictrac'
    and MUST have a single FicTrac .dat file and a corresponding .log file. 
    Returns a single concatenated dataframe. 
    
    The output dataframe is given proper headings, as informed by 
    the documentation on rjdmoore's FicTrac GitHub page. 

    Elapsed time is converted into seconds and minutes, and the integrated 
    X and Y positions are converted to real-world values, by multiplying them 
    against the ball radius. 
    
    Parameters:
    -----------
    root (str): Absolute path to the root directory. I.e. the outermost 
        folder that houses the FicTrac .avi files

    ball_radius (float): The radius of the ball (mm) the insect was on. 
        Used to compute the real-world values in mm.  

    acq_mode (str): The mode with which FicTrac data (.dats and .logs) were 
        acquired. Accepts either 'online', i.e. real-time during acquisition, or 
        'offline', i.e. FicTrac was run after video acquisition.

    do_confirm (bool): If True, prompts the user to confirm the unit of the ball
        radius. If False, skips ths prompt. Default is True. 

    Returns:
    --------
    A list of dataframes. 
    """

    assert acq_mode == "offline" or "online", \
        "Please provide a valid acquisition mode: either 'offline' or 'online'."

    if do_confirm == True:
        confirm = input(f"The ball_radius argument must be in mm. Confirm by inputting 'y'. Otherwise, hit any other key to quit.")
        while True:
            if confirm.lower() == "y":
                break
            else:
                exit("Re-run this function with a ball_radius that's in mm.")
    else:
        pass

    logs = sorted([path.absolute() for path in Path(root).rglob("*.log")])
    dats = sorted([path.absolute() for path in Path(root).rglob("*.dat")])
    
    headers = [ "frame_cntr",
                "delta_rotn_vector_cam_x", 
                "delta_rotn_vector_cam_y", 
                "delta_rotn_vector_cam_z", 
                "delta_rotn_err_score", 
                "delta_rotn_vector_lab_x", 
                "delta_rotn_vector_lab_y", 
                "delta_rotn_vector_lab_z",
                "abs_rotn_vector_cam_x", 
                "abs_rotn_vector_cam_y", 
                "abs_rotn_vector_cam_z",
                "abs_rotn_vector_lab_x", 
                "abs_rotn_vector_lab_y", 
                "abs_rotn_vector_lab_z",
                "integrat_x_posn",
                "integrat_y_posn",
                "integrat_animal_heading",
                "animal_mvmt_direcn",
                "animal_mvmt_spd",
                "integrat_fwd_motn",
                "integrat_side_motn",
                "timestamp",
                "seq_cntr",
                "delta_timestamp",
                "alt_timestamp" ]

    if acq_mode == "online":
        datetimes_from_logs = [get_datetime_from_logs(log) for log in logs]

    dfs = []
    for i, dat in enumerate(dats):
        with open(dat, 'r') as f:
            next(f) # skip first row
            df = pd.DataFrame((l.strip().split(',') for l in f), columns=headers)

        # Convert all values to floats:
        df = df[headers].astype(float)

        # Convert the values in the frame and sequence counters columns to ints:
        df['frame_cntr'] = df['frame_cntr'].astype(int)
        df['seq_cntr'] = df['seq_cntr'].astype(int)
        
        # Compute times and framerate:         
        if acq_mode == "online":
            df["datetime"] = datetimes_from_logs[i]
            df["elapsed"] = df["datetime"][1:] - df["datetime"][0]
            df["secs_elapsed"] = df.elapsed.dt.total_seconds()
            df["framerate_hz"] = 1 / df["datetime"].diff().dt.total_seconds() 

        if acq_mode == "offline":
            # Timestamp from offline acq seems to just be elapsed ms:
            df["secs_elapsed"] = df["timestamp"] / 1000
            df["framerate_hz"] = 1 / df["secs_elapsed"].diff()
        
        df['mins_elapsed'] = df['secs_elapsed'] / 60

        # Discretize minute intervals:
        df['min_int'] = df["mins_elapsed"].apply(np.floor) + 1
        df['min_int'] = df['min_int'].astype(str).str.strip(".0")

        # Compute real-world values:
        df['X_mm'] = df['integrat_x_posn'] * ball_radius
        df['Y_mm'] = df['integrat_y_posn'] * ball_radius
        df['speed_mm_s'] = df['animal_mvmt_spd'] * df["framerate_hz"] * ball_radius

        # Assign ID number:
        df['ID'] = str(i) 

        dfs.append(df)

    return dfs


def parse_dats_by_group(basepath, group_members, 
                        ball_radius, acq_mode, do_confirm):
    
    """
    Parse `.dat` files by a group of manually specified group members. 
    Wraps the `parse_dats()` function.

    Parameters:
    -----------
    basepath: Longest common path shared by each element in `group_members`. 
    group_members: List of group members. Each element must be a substring 
        in the path to the FicTrac `.dat` file. 

    Returns:
    --------
    A list of dataframes
    """

    # assert ("ID" in df), \
    #     "The dataframe must have a column called `ID`"
    # TODO: Checks sorting

    paths = search_for_paths(basepath, group_members)
    dfs = [parse_dats(path, ball_radius, acq_mode, do_confirm) for path in paths]
    
    return flatten_list(dfs)


def process_dats(basepath, group_members, 
                 ball_radius, acq_mode, do_confirm, 
                 cols_to_filter, order, cutoff_freq): 
    
    """
    Process a group of `.dat`s. Reads and parses `.dat`s, adds metadata 
    from corresponding paths, filters specified columns, concatenates 
    dataframes, and regenerates IDs.

    Parameters:
    -----------
    basepath
    group_members
    ball_radius
    acq_mode
    do_confirm
    cols_to_filter: A list of columns to filter

    Returns:
    --------
    A single concatenated dataframe.        
    """

    paths = search_for_paths(basepath, group_members)
    dfs = parse_dats_by_group(basepath, group_members, ball_radius, acq_mode, do_confirm)
    dfs = add_metadata_to_dfs(paths, dfs)

    # If the first row of a column to be filtered is NaN, all subsequent rows are NaNs:
    dfs = [df.dropna() for df in dfs]
    dfs = [bw_filter(df, cols_to_filter, order, cutoff_freq) for df in dfs]
    
    # Filtering results in NaNs in the first row of each dataframe:
    dfs = [df.dropna() for df in dfs] 
    df = regenerate_IDs(pd.concat(dfs))

    return df





def main():
    
    # TODO: Move this documentation to a README.md in software/
    # TODO: Provide the option to turn the svg generations off, bc they take a long time

    # parser = argparse.ArgumentParser(description = __doc__)
    # parser.add_argument("acq_mode", 
    #     help="The mode with which FicTrac data (.dats and .logs) were acquired. \
    #         Accepts either 'online', i.e. real-time during acquisition, or \
    #         'offline', i.e. FicTrac was run after video acquisition.")
    # parser.add_argument("root",
    #     help="Absolute path to the root directory. I.e. the outermost \
    #         folder that houses the FicTrac files.\
    #         E.g. /mnt/2TB/data_in/test/")
    # parser.add_argument("nesting", type=int,
    #     help="Specifies the number of folders that are nested from \
    #         the root directory. I.e. The number of folders between root \
    #         and the 'fictrac' subdirectory that houses the .dat and .log files. \
    #         This subdirectory MUST be called 'fictrac'.")
    # parser.add_argument("ball_radius", type=float,
    #     help="The radius of the ball used with the insect-on-a-ball tracking rig. \
    #         Must be in mm.")
    # # parser.add_argument("val_cols", 
    # #     help="List of column names of the Pandas dataframe to be used as the \
    # #         dependent variables for analyses.")
    # parser.add_argument("time_col",
    #     help="Column name of the Pandas dataframe specifying the time.")
    # parser.add_argument("cmap_col",
    #     help="Column name of the Pandas dataframe specifying the variable to be \
    #         colour-mapped.")
    # parser.add_argument("cutoff_freq", type=float,
    #     help="Cutoff frequency to be used for filtering the FicTrac data.")
    # parser.add_argument("order", type=int,
    #     help="Order of the filter.")
    # parser.add_argument("view_percent", type=float,
    #     help="Specifies how much of the filtered data to plot as an initial \
    #         percentage. Useful for assessing the effectieness of the filter over \
    #         longer timecourses. Default is set to 1, i.e. plot the data over the \
    #         entire timecourse. Must be a value between 0 and 100.")
    # parser.add_argument("percentile_max_clamp", type=float,
    #     help="Specifies the percentile at which to clamp the max depicted \
    #         colourmap values. Plots a span at this value for the population \
    #         histograms and ECDFs.")
    # parser.add_argument("alpha_cmap", type=float,
    #     help="Specifies the transparency of each datum on the XY colourmap plots. \
    #         Must be between 0 and 1, inclusive.")

    # parser.add_argument("val_labels", nargs="?", default=None,
    #     help="list of y-axis label of the generated plots. Default is a formatted \
    #         version of val_cols")
    # parser.add_argument("time_label", nargs="?", default=None,
    #     help="time-axis label of the generated plots. Default is a formatted \
    #         time_col")
    # parser.add_argument("cmap_label", nargs="?", default=None,
    #     help="label of the colour map legend")
    # parser.add_argument("framerate", nargs="?", default=None, type=float,
    #     help="The mean framerate used for acquisition with FicTrac. \
    #         If None, will compute the average framerate. Can be overridden with a \
    #         provided manual value. Default is None.") 
    
    # parser.add_argument("-ns", "--no_save", action="store_true", default=False,
    #     help="If enabled, does not save the plots. By default, saves plots.")
    # parser.add_argument("-sh", "--show", action="store_true", default=False,
    #     help="If enabled, shows the plots. By default, does not show the plots.")
    # args = parser.parse_args()

    # TODO: Write docs for script arguments, maybe in a README.md
    # TODO: Don't hardcode .yaml file name, pass it in as an argument instead.
    # TODO: Specify default .yaml values, for key-value pairs that are unspecified:
    with open("fictrac_analyses_params.yaml") as f:

        params = yaml.load(f, Loader=yaml.FullLoader)
        # print(params)

    root = params["root"]
    acq_mode = params["acq_mode"]
    acq_mode = params["acq_mode"]
    acq_mode = "offline"
    ball_radius = params["ball_radius"]

    val_cols = params["val_cols"]
    filtered_val_cols = ["filtered_" + val_col for val_col in val_cols]
    val_labels = params["val_labels"]

    time_col = params["time_col"]
    time_label = params["time_label"]

    cutoff_freq = params["cutoff_freq"]
    order = params["order"]
    framerate = params["framerate"]

    view_perc = params["view_perc"]

    cmap_labels = params["cmap_labels"]
    alpha_cmap = params["alpha_cmap"]
    percentile_max_clamp = params["percentile_max_clamp"]
    respective = params["respective"]
    respective = False

    no_save = params["no_save"]
    show_plots = params["show_plots"]

    # Parse FicTrac inputs:
    dfs = parse_dats(root, ball_radius, acq_mode, do_confirm=False)

    # Save each individual bokeh plot to its respective ID folder. 
    folders = sorted([path.absolute() for path in Path(root).rglob("*/fictrac")])

    save_paths = []
    for df, folder in zip(dfs, folders):
        # Generate individual ID plots:
        print(f"Generating individual plots for {folder} ...")
        save_path_to = join(folder, "plots/")
        save_paths.append(save_path_to)

        if exists(save_path_to):
            rmtree(save_path_to)
        mkdir(save_path_to)

        if no_save == True:
            save_path_to = None

        # Plot FFT power spectrum:
        viz.plot_fft(df, 
                 val_cols=val_cols, 
                 time_col=time_col, 
                 val_labels=val_labels,
                 time_label=time_label,
                 cutoff_freq=cutoff_freq, 
                 save_path_to=save_path_to,
                 show_plots=show_plots) 

        # Plot raw vs. filtered:
        viz.plot_filtered(df, 
                              val_cols=filtered_val_cols, 
                              time_col=time_col, 
                              val_labels=val_labels, 
                              time_label=time_label,
                              cutoff_freq=cutoff_freq, 
                              order=order, 
                              view_perc=view_perc,
                              save_path_to=save_path_to,
                              show_plots=show_plots)

        # Plot XY
        cm = viz.get_all_cmocean_colours()
        viz.plot_trajectory(df,
                        cmap_cols=filtered_val_cols,
                        high_percentile=percentile_max_clamp,
                        respective=respective,
                        cmap_labels=cmap_labels,
                        palette=cm["thermal"],
                        alpha=alpha_cmap,
                        save_path_to=save_path_to,
                        show_plots=show_plots)

    # Generate population plots:
    print("Generating population plots ...")

    # Concatenate data into a population:
    concat_df = pd.concat(dfs).dropna()
    
    save_path_popns = join(root, "popn_plots/")
    if exists(save_path_popns):
        rmtree(save_path_popns)
    mkdir(save_path_popns)
    subdirs = ["histograms/", "ecdfs/"]
    [mkdir(join(root, "popn_plots/", subdir)) for subdir in subdirs]

    save_path_hists = join(root, "popn_plots/", "histograms/")
    save_path_ecdfs = join(root, "popn_plots/", "ecdfs/")

    # Plot histograms:
    print("Generating histograms ...")
    viz.plot_histograms(concat_df, 
                    cutoff_percentile=percentile_max_clamp,
                    save_path_to=save_path_hists, 
                    show_plots=False)

    # Plot ECDFs:
    print("Generating ECDFs ...")
    viz.plot_ecdfs(concat_df,
               cutoff_percentile=percentile_max_clamp,
               save_path_to=save_path_ecdfs, 
               show_plots=False)
    

if __name__ == "__main__":
    main()