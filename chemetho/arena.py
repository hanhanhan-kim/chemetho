from pathlib import Path
from os.path import basename, splitext
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

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


def prefix(x):
    
    """Get the prefix of a path (str), where the prefix MUST end with 
    yyyy-mm-dd, and underscores delimit everything else."""
    
    date_fmt_str = "%Y-%m-%d"
    delim = '_'

    parts = splitext(x)[0].split(delim)

    prefix_parts = []
    for x in parts:
        try:
            prefix_parts.append(x)
            _ = datetime.strptime(x, date_fmt_str)
            break
        except ValueError:
            continue

    return delim.join(prefix_parts)


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
    
    if csv_prefix != txt_prefix != circ_pkl:
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


