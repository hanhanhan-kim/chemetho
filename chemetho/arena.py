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


def merge_arena_data(dlc_csv, times_txt, circ_pkl):
    
    """
    Merge DeepLabCut tracking data (.csv), raspivid timestamps (.txt), and 
    circular arena (.pkl) dimensions, into a single dataframe. 
    
    Parameters:
    -----------
    dlc_csv (str): Path to .csv output of DeepLabCut tracking results. 
    times_txt (str): Path to .txt output of raspivid timestamps. 
    circ_pkl (str): Path to .pkl output of `vidtools find-circle` command. 
    
    Returns:
    --------
    A Pandas dataframe. 
    """
    
    if splitext(dlc_csv)[1] != ".csv":
        raise ValueError("`dlc_csv` must end in '.csv'")
    if splitext(times_txt)[1] != ".txt":
        raise ValueError("`times_txt` must end in '.txt'")
    if splitext(circ_pkl)[1] != ".pkl":
        raise ValueError("`circ_pkl` must end in '.pkl'")
    
    csv_prefix = basename(prefix(dlc_csv)).strip()
    txt_prefix = basename(prefix(times_txt)).strip()
    pkl_prefix = basename(prefix(circ_pkl)).strip()
    
    if csv_prefix != txt_prefix != pkl_prefix:
        raise ValueError("The filenames of `dlc_csv`, `times_txt`, and `circ_pkl` " 
                         "suggest they are not from the same experiment. "
                         f"\ndlc_csv prefix: {csv_prefix}"
                         f"\ntimes_txt prefix: {txt_prefix}"
                         f"\ncirc_pkl prefix: {pkl_prefix}")
    
    behav = parse_dlc_csv(dlc_csv)
    times = parse_raspivid_times(times_txt)
    
    if len(behav) != len(times):
        raise ValueError("The lengths of `dlc_csv` and `times_txt` do not match.")
    
    # Merge on indexes:
    merged = pd.merge(behav, times, left_index=True, right_index=True)
    
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