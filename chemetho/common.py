from os.path import join, commonpath
from pathlib import Path
from functools import reduce
import dateutil

from pandas.api.types import is_numeric_dtype
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
    basepath (str): Longest common path shared by each glob search return. 
    group_members (list): List of group members. Each element must be a substring
        in the path to the `glob_ending` result. 
    glob_ending (str): A Unix-style path ending. Supports '*' wildcards. 

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
    assert len(paths) == len(dfs), "Lengths of `paths` and `dfs` are unequal"
    # TODO: Add check to see if `paths` and `dfs` are sorted in the same way. 

    dfs_with_metadata = []
    for path, df in zip(paths, dfs):

        splitted = path.split("/")

        for i, folder in enumerate(splitted):
            try:
                dateutil.parser.parse(folder)
                date = path.split("/")[i]
                animal = path.split("/")[i+1]
                trial = path.split("/")[i+2]
                break
            except:
                pass
        
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


def process_csvs(basepath, group_members, glob_ending="*/stimulus/*.csv"):
    
    """
    Process a group of `.csv`s. Reads the `.csv`s, adds metadata from 
    corresponding paths, concatenates dataframes, and regenerates IDs.

    Assumes that there exists somewhere after the 'basepath', a directory 
    structure that goes 'date -> animal -> trial', where 'trial' holds the 
    `.csv` data. 

    This function does not perform any filtering. 

    Parameters:
    -----------
    basepath (str): Longest common path shared by each glob search return. 
    group_members (list): List of group members. Each element must be a substring
        in the path to the `glob_ending` result. 
    glob_ending (str): A Unix-style path ending. Supports '*' wildcards. 

    Returns:
    --------
    """

    paths = search_for_paths(basepath, group_members, glob_ending=glob_ending)
    dfs = [pd.read_csv(path) for path in paths]
    dfs = add_metadata_to_dfs(paths, dfs)
    df = regenerate_IDs(pd.concat(dfs))

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
    # TODO: 
    
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


def get_smaller_last_val(df_1, df_2, common_col):

    """
    Compares the last value for a column shared between two dataframes. 

    Parameters:
    ------------
    df_1: A dataframe

    df_2: A dataframe

    common_col: A shared column for comparing 'df_1' against 'df_2'. 
    
    Returns:
    ---------
    The smaller value of these values as a float.  
    """

    assert(common_col in df_1 and common_col in df_2), f"df_1 and df_2 do not share {common_col}"
    assert(is_numeric_dtype(df_1[common_col])), f"The values of {common_col} in df_1 is not numeric, e.g. float64, etc."
    assert(is_numeric_dtype(df_2[common_col])), f"The values of {common_col} in df_1 is not numeric, e.g. float64, etc."

    df_1_is_bigger = float(df_1[common_col].tail(1)) > float(df_2[common_col].tail(1))

    if df_1_is_bigger is True:
        return float(df_2[common_col].tail(1))
    else:
        return float(df_1[common_col].tail(1))


def get_common_expts(df_1, df_2):
    
    """
    Returns the date + animal + trial combinations common to both 'df_1' and 'df_2'. 
    Prints out the date + animal + trial combinations unique to 'df_1' or 'df_2'. 
    
    Parameters:
    -----------
    df_1: A dataframe to be merged with 'df_2'. 
    df_2: A dataframe to be merged with 'df_1'.
    
    Returns:
    --------
    A list of tuples, where each tuple has a condition common to both input dataframes. 
    """
    
    grouped_1 = df_1.groupby(["date", "animal", "trial"])
    grouped_2 = df_2.groupby(["date", "animal", "trial"])
    
    names_1 = {name for name,_ in grouped_1}
    names_2 = {name for name,_ in grouped_2}
    
    in_df_1_only = names_1.difference(names_2)
    in_df_2_only = names_2.difference(names_1)
    
    if len(in_df_1_only) > 0:
        print(f"The following conditions are unique to 'df_1': {in_df_1_only} \n")
    else:
        print("'df_1' describes a subset of 'df_2' experiments \n")
        
    if len(in_df_2_only) > 0:
        print(f"The following conditions are unique to 'df_1': {in_df_2_only} \n")
    else:
        print("'df_2' describes a subset of 'df_1' experiments \n")
        
    if len(in_df_1_only) == 0 and len(in_df_1_only) == 0:
        print("'df_1' and 'df_2' describe identical experiments \n")
        
    in_both_dfs = names_1.intersection(names_2)
    
    return list(in_both_dfs)
    

def merge_timeseries(df_1, df_2, 
                     common_time="secs_elapsed", 
                     fill_method="ffill"):
    
    """
    Merges, according to a common column, two ordered dataframes, e.g. timeseries. 
    Truncates the merged dataframe by the earliest last valid input observation. 
    Merges only those experiments (i.e. a unique date + animal + trial combo) 
    common to both dataframes.

    Parameters:
    ----------- 
    df_1: A time-series dataframe to be merged with 'df_2'.

    df_2: A time-series dataframe to be merged with 'df_1'. 

    common_time (str): A common column against which to merge the dataframes. 
        Must be some ordered unit such as time. 

    fill_method (str): Specifies how to treat NaN values upon merge. 
        Either 'ffill' for forward fill, or 'linear' for linear interpolation. 
        Forward fill fills the NaNs with the last valid observation, until the
        next valid observation. Linear interpolation fits a line based on two 
        flanking valid observations. The latter works only on columns with numeric 
        values; non-numerics are forward-filled. 

    Returns:
    --------
    A Pandas dataframe. 
    """
    
    assert(common_time in df_1 and common_time in df_2), f"'df_1' and df_2 do not share {common_time}"
    assert("date" in df_1), "The column, 'date' is not in 'df_1'"
    assert("animal" in df_1), "The column, 'animal' is not in 'df_1'"
    assert("trial" in df_1), "The column, 'trial' is not in 'df_1'"
    assert("date" in df_1), "The column, 'date' is not in 'df_2'"
    assert("animal" in df_1), "The column, 'animal' is not in 'df_2'"
    assert("trial" in df_1), "The column, 'trial' is not in 'df_2'" 

    if "ID" in df_1:
        df_1.drop("ID", axis=1, inplace=True)
    if "ID" in df_2:
        df_2.drop("ID", axis=1, inplace=True)

    # Extract only those data common to both input dataframes:
    in_both_dfs = get_common_expts(df_1, df_2)

    grouped_1 = df_1.groupby(["date", "animal", "trial"])
    grouped_2 = df_2.groupby(["date", "animal", "trial"])
    common_data_1 = [group for name,group in grouped_1 if name in in_both_dfs]
    common_data_2 = [group for name,group in grouped_2 if name in in_both_dfs]

    assert(len(common_data_1) == len(common_data_2)), "Lengths of common_data_1 and common_data_2 are unequal"

    merged_dfs = []
    for common_df_1, common_df_2 in zip(common_data_1, common_data_2):

        # Compare common_df_1 vs common_df_2:
        smaller_last_val = get_smaller_last_val(common_df_1, common_df_2, common_time)

        if fill_method is "ffill":

            # Merge common_df_1 with common_df_2:
            merged_df = pd.merge_ordered(common_df_1, common_df_2, 
                                        on=[common_time, "date", "animal", "trial"], 
                                        fill_method=fill_method)

            # Truncate merged with smaller of the mergees:
            merged_df = merged_df.loc[merged_df[common_time] <= smaller_last_val]   
        
        elif fill_method is "linear":

            # Merge common_df_1 with common_df_2:
            merged_df = pd.merge_ordered(common_df_1, common_df_2, 
                                        on=[common_time, "date", "animal", "trial"], 
                                        fill_method=None)

            # Truncate merged with smaller of the mergees:
            merged_df = merged_df.loc[merged_df[common_time] <= smaller_last_val]

            # Finally interpolate:
            merged_df = merged_df.interpolate(method=fill_method)

            # Ffill any remaining non-numeric values:
            merged_df = merged_df.ffill(axis=0)

        merged_dfs.append(merged_df)

    final_df = pd.concat(merged_dfs)
    final_df = regenerate_IDs(final_df)

    return final_df


# TODO: have this fxn wrap around my merge_timeseries() fxn! 
def merge_n_ordered(dfs, on, fill_method, truncate_on=None):
    
    """
    Merge n number of ordered dataframes. 
    
    Parameters:
    -----------
    dfs: List of ordered dataframes
    on (str): Common column across `dfs` by which to merge.
    fill_method (str): Currently only accepts "ffill" #TODO: Add linear interpolation option
    truncate_on (fl): Value in `on` column to which the dataframe will be truncated. 
        If None, will not truncate values in `on` column. Default is None.
    
    Returns:
    --------
    A merged dataframe. 
    """
    
    # TODO: Check that the value for truncate_on is a value within the range of the on column:
    
    fxn = lambda left,right: pd.merge_ordered(left,right,on=on, fill_method=fill_method)
    reduced_df = reduce(fxn, dfs)
    
    # TODO: Here, 'truncate_on' is an argument. By calling merge_timeseries() I'll free up this param, i.e. won't have to do it: 
    if truncate_on is not None:
        return reduced_df.loc[reduced_df[on] <= truncate_on]
    
    return reduced_df


def cut_before_1st_instance(df, col, val):

    """
    Cuts the dataframe before the 1st instance of a specified value in a column.

    Parameters:
    -----------
    df: A Pandas dataframe.
    col (str): A column in the Pandas dataframe
    val (num):

    Returns:
    --------
    A Pandas dataframe 
    """

    assert(col in df), f"The dataframe does not have the column, {col}"

    neck = df.loc[df[col]==val].index[0] 
    beheaded_df = df[neck:]
    
    return beheaded_df