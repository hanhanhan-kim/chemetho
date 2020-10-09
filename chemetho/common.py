from os.path import join, commonpath
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.signal as sps

from .constants import banned_substrings

# TODO: Check for dataframe object type:

def unconcat(concat_df, col_name="ID"):

    """
    Splits up a concatenated dataframe according to each unique `col_name`.
    Returns a list of datafrmaes. 

    Parameters:
    -----------
    concat_df: A Pandas dataframe
    col_name (str): A column name in 'concat_df' with which to split into smaller dataframes. 
        Default is "ID". 

    Returns:
    --------
    A list of dataframes, split up by each `col_name`. 
    """

    assert (col_name in concat_df), \
        f"The column, {col_name}, is not in in the input dataframe."

    dfs_by_ID = []

    for df in concat_df[col_name].unique():
        df = concat_df.loc[concat_df[col_name]==df]
        dfs_by_ID.append(df)

    return(dfs_by_ID)


def flatten_list(list_of_lists):
    
    """
    Flatten a list of lists into a list.

    Parameters:
    -----------
    list_of_lists: A list of lists

    Returns:
    --------
    A list.
    """
    
    # Reach into each inner list and append into a new list: 
    flat_list = [item for inner_list in list_of_lists for item in inner_list]

    return flat_list


def ban_columns_with_substrings(df, substrings=banned_substrings):

    """
    From a dataframe, return only columns whose names do not contain specified substrings.

    Parameters:
    -----------
    df: A dataframe. 
    substrings: A list of substrings to ban from `df`'s column names.

    Returns:
    --------
    A list of all of `df`'s column names that do not contain the substrings listed in `substrings`.
    """
    
    all_cols = list(df.columns)
    
    banned_cols = []
    for col in all_cols:
        for substring in banned_substrings:
            if substring in col:
                banned_cols.append(col)
                break

    ok_cols = [col for col in all_cols if col not in banned_cols]

    return ok_cols


def read_csv_and_add_metadata(paths):

    """
    Reads `.csv` data into a dataframe, then adds metadata from each file's path to the 
    dataframe. Assumes that there exists somewhere in the path, a directory structure 
    that goes 'date -> animal -> trial', where 'trial' holds the `.csv` data. 
    
    Parameters:
    ------------
    paths: A list of paths

    Returns:
    ---------
    A list of dataframes
    """
    
    common_path = commonpath(paths)
    
    dfs = []
    for path in paths:
        
        new_path = path.replace(f"{common_path}/", "")
        date = new_path.split("/")[0]
        animal = new_path.split("/")[1]
        trial = new_path.split("/")[2]
        
        df = pd.read_csv(path)
        df["date"] = date
        df["animal"] = animal
        df["trial"] = trial
        
        dfs.append(df)
        
    return(dfs)


def search_for_paths(basepath, group_members, glob_ending="*/fictrac"):

    """
    Search with a list of subdirectories, to get a list of paths, 
    where each path ends in `glob_ending`. 

    Parameters:
    -----------
    basepath: Longest common path shared by each glob search return. 
    group_members: List of group members. Each element must be a substring
        in the path to the `glob_ending` result. 
    glob_ending: A Unix-style path ending. Supports '*' wildcards. 

    Returns:
    --------
    A list of paths
    """

    # TODO: Fix--currently assumes that group_member IMMEDIATELY follows basepath, 
    # as seen in the join() below. Need to make it more general. 
    
    new_paths = [str(path.absolute()) 
                for group_member in group_members 
                for path in Path(join(basepath, group_member)).rglob(glob_ending)]
    
    return sorted(new_paths)


def add_metadata_to_dfs(paths, dfs):

    """
    Adds metadata from each file's path to the dataframe. Assumes that there exists 
    somewhere in the path, a directory structure that goes 'date -> animal -> trial', where
    'trial' holds the data, e.g. `.dat` or `.csv`. 
    
    Parameters:
    -------------
    paths: list of paths
    dfs: list of dataframes

    Returns:
    ---------
    A list of dataframes
    """

    # TODO: How to add a check for "date -> animal -> trial -> .dat" structure? 

    common_path = commonpath(paths)
    
    dfs_with_metadata = []
    for path, df in zip(paths, dfs):
        
        assert len(paths) == len(dfs), "Lengths of `paths` and `dfs` are unequal"
        # TODO: Add check to see if `paths` and `dfs` are sorted in the same way. 
        
        new_path = path.replace(f"{common_path}/", "")
        date = new_path.split("/")[0]
        animal = new_path.split("/")[1]
        trial = new_path.split("/")[2]
        
        df["date"] = date
        df["animal"] = animal
        df["trial"] = trial
        
        dfs_with_metadata.append(df)
        
    return(dfs_with_metadata)


def regenerate_IDs(df, group_by=["date", "animal", "trial"]):

    """
    Regenerate unique animal IDs according to some groupby criteria. 
    Useful for when a previous function sets IDs according to 
    some intermediate groupby structure. 
    
    Parameters:
    ------------
    df: dataframe
    group_by: a list of columns in `df` with which to perform the groupby
    
    Returns:
    ---------
    A dataframe
    """

    # TODO: Check that elements of group_by are columns in the dataframe

    # Assign unique group IDs:
    df["ID"] = (df.groupby(group_by).cumcount()==0).astype(int)
    df["ID"] = df["ID"].cumsum()
    df["ID"] = df["ID"].apply(str)
    
    return df


def curate_by_date_animal(df, included):

    """
    Curate a dataframe according to values in its `date` and `animal` columns.

    Parameters:
    ------------
    df: the dataframe to be curated
    included: list of (<date>, <animal>) tuples to be included in the groupby
    
    Returns:
    ---------
    The curated dataframe and the number of animals after curation
    """

    assert ("date" in df), "The dataframe must have a column called 'date'"
    assert ("animal" in df), "The dataframe must have a column called 'animal'"
    
    # groupby has 3 keys:
    grouped = df.groupby(["date", "animal", "trial"])
    
    # Apply curation to generate a list of dataframes:
    groups = []
    for name, group in grouped:
        # Slice based on first 2 keys, date and animal, only:
        if name[:2] in included:
            groups.append(group)
            
    # Concatenate the dataframes back together:
    concat_curated_df = pd.concat(groups)
    
    # Get n animals:
    n_animals = len(concat_curated_df.groupby(["date", "animal"]))
    
    return(concat_curated_df, n_animals)


def baseline_subtract(df, baseline_end, time_col, val_col):
    
    # TODO: Update val_col to val_cols for multiple vals:
    
    """
    Computes a mean baseline value for a column in the dataframe 
    and subtracts that value from the data.
    
    Parameters:
    -----------
    df: A dataframe
    baseline_end (fl): The time at which the baseline period ends.
    time_col:  
    val_col:
    
    Returns:
    --------
    A dataframe
    """
    
    # Compute a mean value as the baseline:
    baseline = np.mean(df.loc[df[time_col] < baseline_end][val_col])

    # Subtract baseline from val_col:
    df[val_col] = df[val_col] - baseline
    
    return df


def compute_z_from_subseries(series, subseries):
    
    """
    From timeseries data, compute the z-score based on a mu and sigma 
    that are derived from a subset of the timeseries (a subseries), 
    e.g. from a pre-stimulus period. 

    Parameters:
    -----------
    
    
    Returns:
    --------
    A z-score for each datapoint in the entire timeseries.
    """
    
    mu = np.mean(subseries)
    sigma = np.std(subseries) 
    
    return ([(val - mu) / sigma for val in series])


def compute_z_from_subdf(df, val_col, time_col, 
                         subseries_end, subseries_start=0):
    """
    From a timeseries dataframe, compute the z-score based on a mu 
    and sigma that are derived from a subset of the timeseries (a 
    sub-dataframe), e.g. from a pre-stimulus period. 

    Parameters:
    -----------

    
    Returns:
    --------
    A z-score for each datapoint in the entire timeseries dataframe.
    """
    
    # Separate out the features:
    x = df[val_col]

    # Standardize the features:
    start = df[time_col] > subseries_start
    end = df[time_col] < subseries_end
    
    x_sub = df.loc[start & end][val_col]
    x = compute_z_from_subseries(x, x_sub)

    # Add the feature back to the dataframe:
    # TODO: Remove (sth) before adding (z-score)
    df[f"{val_col} (z-score)"] = x
    
    return df


def bw_filter(df, val_cols, order, cutoff_freq, framerate=None):

    """
    Applies low-pass Buterworth filter on offline FicTrac data. 
    Does not drop NA values.

    Parameters:
    -----------
    df (DataFrame): Dataframe of FicTrac data generated from parse_dats()

    val_cols (list): List of column names from `df` to be filtered. 

    order (int): Order of the filter.

    cutoff_freq (float): The cutoff frequency for the filter in Hz.

    framerate (float): The mean framerate used for acquisition with FicTrac. 
        If None, will use the average frame rate as computed in the input 'df'. 
        Can be overridden with a provided manual value. Default is None.

    Returns:
    --------
    A dataframe with both the filtered and raw values. 
    Filtered columns are denoted with a "filtered_" prefix.
    """
        
    if framerate == None:
        framerate = np.mean(df["framerate_hz"]) 
    
    all_filtered_vals = []
    filtered_cols = []

    for val_col in val_cols:

        vals = list(df[str(val_col)])

        # Design low-pass filter:
        b, a = sps.butter(int(order), float(cutoff_freq), fs=framerate)
        # Apply filter and save:
        filtered_vals = sps.lfilter(b, a, vals)
        all_filtered_vals.append(filtered_vals)

        filtered_col = f"filtered_{val_col}"
        filtered_cols.append(filtered_col)
    
    # Convert filtered values into df:
    filtered_cols_vals = dict(zip(filtered_cols, all_filtered_vals))
    filtered_df = pd.DataFrame.from_dict(filtered_cols_vals)
    df_with_filtered = pd.concat([df, filtered_df], axis=1)

    return df_with_filtered


def aggregate_trace(df, group_by, method="mean", round_to=0, f_steps=1):
    
    """
    From a dataframe with time series data, round the data, do a groupby,
    compute either the mean or the median, then reset the index. 

    Parameters:
    ------------
    df: A Pandas dataframe.
    group_by: A list of columns in `df` with which to perform the groupby. 
    method: The method by which to aggregate the data. Must be either 
        "mean" or "median". Is "mean" by default. 
    round_to: The place value with which to round the data. 
    f_steps (fl): The fraction of steps from which to downsample `df`. 

    Return:
    -------
    A dataframe of aggregate statistics. 
    """

    assert (method=="mean" or method=="median"), \
        "The aggregation `method` must be 'mean' or 'median'."
    assert (1 >= f_steps > 0), \
        f"`f_steps`, {f_steps}, must be greater than 0 and less than or equal to 1."
    
    # Round:
    rounded = df.round(round_to)

    # Downsample:
    indices = np.round(np.linspace(0, len(rounded.index), int(len(rounded.index)*f_steps + 1)))
    indices = [int(index) for index in indices]
    indices.pop()
    downsampled = rounded.iloc[indices,:]
    grouped = downsampled.groupby(group_by) 

    # Aggregate:
    if method=="mean":
        mean_df = grouped.mean().reset_index()
        return mean_df

    elif method=="median":
        median_df = grouped.median().reset_index()
        return median_df