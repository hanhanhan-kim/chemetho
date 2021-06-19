from pathlib import Path
from os.path import basename, splitext
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from .common import prefix


def parse_dlc_csv(csv):
    
    """
    Parses the .csv tracking output from DeepLabCut into a dataframe.
    
    Parameters:
    -----------
    csv (str): Path to csv of DeepLabCut tracking outputs. 
    
    Returns:
    --------
    A Pandas datafame. 
    """
        
    df = pd.read_csv(csv, header=[0,1,2])
    df.columns = df.columns.get_level_values(1) \
                 + ['_' for col in df.columns] \
                 + df.columns.get_level_values(2)
    df = df.drop(columns=["bodyparts_coords"]) # this col repeats the indices
        
    return df


def parse_raspivid_times(txt):
    
    """
    Parses the .txt timestamps from a raspivid recording into a dataframe.
    
    Parameters:
    -----------
    
    Returns:
    --------
    A Pandas dataframe.
    """
    
    df = pd.read_csv(txt)
    df.columns = ["time (s)"]
    df["time (s)"] = df["time (s)"] / 1000
    
    return df


def merge_arena_data(dets_csv, times_txt, circ_pkl):
    
    """
    Merge Detectools (Detectron2 Faster R-CNN) tracking data (.csv), 
    raspivid timestamps (.txt), and circular arena (.pkl) dimensions, 
    into a single dataframe. 
    
    Parameters:
    -----------
    dets_csv (str): Path to .csv output of Detectools tracking results. 
    times_txt (str): Path to .txt output of raspivid timestamps. 
    circ_pkl (str): Path to .pkl output of `vidtools find-circle` command. 
    
    Returns:
    --------
    A Pandas dataframe. 
    """
    
    if splitext(dets_csv)[1] != ".csv":
        raise ValueError("`dets_csv` must end in '.csv'")
    if splitext(times_txt)[1] != ".txt":
        raise ValueError("`times_txt` must end in '.txt'")
    if splitext(circ_pkl)[1] != ".pkl":
        raise ValueError("`circ_pkl` must end in '.pkl'")
    
    csv_prefix = basename(prefix(dets_csv)).strip()
    txt_prefix = basename(prefix(times_txt)).strip()
    pkl_prefix = basename(prefix(circ_pkl)).strip()
    
    if csv_prefix != txt_prefix != pkl_prefix:
        raise ValueError("The filenames of `dets_csv`, `times_txt`, and `circ_pkl` " 
                         "suggest they are not from the same experiment. "
                         f"\ndets_csv prefix: {csv_prefix}"
                         f"\ntimes_txt prefix: {txt_prefix}"
                         f"\ncirc_pkl prefix: {pkl_prefix}")
    
    behav = pd.read_csv(dets_csv)
    times = parse_raspivid_times(times_txt)
    times = times.reset_index().rename(columns={"index": "frame"})
    
    if len(behav["frame"].unique()) != len(times):
        raise ValueError("The number of frames in `dets_csv` and `times_txt` do not match.")
    
    # Merge on indexes:
    merged = pd.merge_ordered(behav, times, on="frame")
    
    circ = pickle.load(open(circ_pkl, "rb"))
    
    merged["circle_centre_x"] = circ["x (pxls)"]
    merged["circle_centre_y"] = circ["y (pxls)"]
    merged["circle_radius"] = circ["r (pxls)"] 
    
    return merged


def get_dist(a_x, a_y, b_x, b_y):

    """Get the distance between two Cartesian points, where each coordinate is an argument."""

    return np.sqrt( ((a_x-b_x)**2) + ((a_y-b_y)**2) )


def get_speed(df, x_col, y_col, t_col):
    
    """
    Compute the speed from a dataframe with x-coord, y-coord, and time columns.
    
    Parameters:
    -----------
    df: A Pandas dataframe.
    x_col (str): The name of the column for the x-coordinate.
    y_col (str): The name of the column for the y-coordinate.
    t_col (str): The name of the column for the t-coordinate. 
    
    Returns:
    --------
    A Pandas series.
    """
    
    if not (x_col in df and y_col in df and t_col in df):
        raise ValueError(f"All 3 columns ({x_col}, {y_col}, and {t_col}) must be in df")
    
    speed = np.sqrt(df[x_col].diff()**2 + df[y_col].diff()**2) / df[t_col].diff() 
    
    return speed 


def get_centroid_from_bbox(x1, y1, x2, y2):

    """
    Compute the centroid of a bounding box. 
    Recall that in OpenCV, the image origin is the top left corner. 
    
    Parameters:
    -----------
    x1 (fl): x-coord of top left bounding box corner.
    y1 (fl): y-coord of top left bounding box corner.
    x2 (fl): x-coord of bottom right bounding box corner.
    y2 (fl): y-coord of bottom right bounding box corner. 

    Returns:
    --------
    x-coord of centroid and y-coord of centroid
    """
    
    # # Won't work if arguments are Pandas series:
    # if x2 < x1:
    #     raise ValueError(f"x1, {x1}, must be less than x2, {x2}")
    # if y2 < y1:
    #     raise ValueError(f"y1, {y1}, must be less than y2, {y2}")

    centre_x = (x2 - x1)/2 + x1
    centre_y = (y2 - y1)/2 + y1

    return centre_x, centre_y


def get_intxn_bouts(df, dist_col="dist_bw_animals (mm)", closeness=0.1, window=30):
    
    """
    Get interaction bouts between 2 animals. Assumes that
    only 2 animals are in the video. Will not capture overlapping 
    interaction bouts.
    
    Parameters:
    -----------
    df: An ordered Pandas dataframe, e.g. timeseries, of a single 
        animal pair. 
    dist_col (str): The column in `df` that specifies the 
        distance between the 2 animals.
    closeness (float): The criterion for defining whether the 
        animals were close. Is a percentile of the distance 
        between animals. Must be a value between 0 and 1. 
    window (int): The number of rows to look at before and after 
        the animals were close together. 
        
    Returns:
    --------
    A list of dataframes, where each dataframe is an interaction bout. 
    """
    
    if dist_col not in df:
        raise ValueError(f"'{dist_col}' is not a column in the dataframe.")
        
    if closeness > 1 or closeness < 0:
        raise ValueError(f"The value for closeness, {closeness}, must be between 0 and 1.")
    
    # Set a cutoff for a close interaction:
    dist_cutoff = np.percentile(df["dist_bw_animals (mm)"].dropna(), 0.1)
    
    # Get indices of close interaction instances:
    dist_crit = df["dist_bw_animals (mm)"] < dist_cutoff
    idxs = df.loc[dist_crit].index

    # A lot of these idxs are just +1 from the previous idx, which is redundant, because when
    # I then grab the next e.g. 100 rows, 99 of the rows will have overlapped. 
    # I need to get only the non-overlapping blocks of rows idxs:
    window = 30
    old_idx = 0
    new_idxs = []
    for idx in idxs:
        if idx - old_idx > window:
            new_idxs.append(idx)
        old_idx = idx

    # Get moments before and after each episode of closeness:
    bouts = [df.iloc[idx-window : idx+window] for idx in new_idxs] # list of dfs
    
    # Have to readjust time, so it's centred around the interaction point: 
    for bout in bouts:
        bout["zeroed_time (s)"] = bout["time (s)"] - bout.iloc[window]["time (s)"]
    
    return bouts