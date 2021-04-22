import pandas as pd
import numpy as np

from scipy.signal import find_peaks, peak_widths


def get_counts_df(df, method="full", do_relative=False):
    
    """
    Converts a pyteeomics.mzxml -> dataframe of Shimadzu GC-MS data into
    a dataframe of counts. If relative, the raw count data is 
    normalized by the total number of counts. 
    
    Parameters:
    -----------
    df: A dataframe generated from pyteomics.mzxml.read()
    method (str): Specifies whether the GC-MS data was collected via a SIM 
        (selective ion method) or a full spectra method. Must be 
        either "SIM" or "full". Default is "full".
    do_relative (bool): If True, will normalize the raw count data by the 
        total number of counts. If False, will not normalize. Default is 
        False.  
    
    Returns:
    --------
    A dataframe of mass/charge, retention time, and relative intensity
    """
    
    if method == "SIM".lower():

        dfs = []
        
        grouped = df.groupby(['lowMz'])
        for _, group in enumerate(grouped):
            
            # group[0] holds the m/z:
            low_mz = f"{group[0]}"
            
            # group[1] holds the actual dataframe:
            ret_times = group[1]['retentionTime']
            
            raw = [np.sum(val) for val in group[1]['intensity array'].values]
            denom = np.sum(group[1]['intensity array'].values)

            rel = raw / denom
            
            if do_relative:

                rel_df = pd.DataFrame.from_dict({"mass/charge": low_mz,
                                                    "retention time (mins)": ret_times, 
                                                    "relative counts": rel}
                                                    ).reset_index(drop=True)
                
                dfs.append(rel_df)

            else:

                raw_cnts_df = pd.DataFrame.from_dict({"mass/charge": low_mz,
                                                    "retention time (mins)": ret_times, 
                                                    "counts": raw}
                                                    ).reset_index(drop=True)
                
                dfs.append(raw_cnts_df)

        return(pd.concat(dfs))

    elif method == "full".lower():

        raw = np.array([np.sum(row) for row in df["intensity array"]])
        denom = np.sum(raw)
        rel = raw / denom

        if do_relative:
        
            df["relative counts"] = rel

        else:

            df["counts"] = raw
        
        return df

    else:
        raise ValueError("The `method` argument must be 'SIM' or 'full'.")


def get_peaks(df, val_col, threshold, width_from_top=0.98):

    """
    
    Parameters:
    -----------
    df: A dataframe generated from `get_counts_df()`.
    val_col (str): A column name in `df` for GC-MS counts data. 
        E.g. 'relative counts' or 'counts'. 
    threshold (fl): The threshold value for peak-calling. Peak shapes above this 
        value are classified as peaks. 
    width_from_top (fl): A value between 0 and 1 that specifies how far from the 
        top of the peak to call the peak width. Higher values are further from 
        the peak.  

    Returns:
    --------
    A dataframe with the following columns:
        peaks_times: The peaks' retention times. 
        peaks_y: The y-axis values for the peaks. 
        left_times: The left side of the peak width, in retention times. 
        right_times: The right side of the peak width, in retention times.  
        peaks_width_y: The y-axis values for the peak widths. Both the left 
            and right sides of the peak width have the same y-values.
    """

    if "retentionTime" not in df:
        raise ValueError(f"'retentionTime' is not a column in the dataframe")
    if val_col not in df:
        raise ValueError(f"'{val_col}' is not a column in the dataframe")

    data = df[val_col]

    # baseline = peakutils.baseline(data)
    peak_idxs,_ = find_peaks(data, height=threshold)
    widths = peak_widths(data, peak_idxs, rel_height=width_from_top)
    width_dists = widths[1]
    # peak_widths() interpolates idxs, so I have to round:
    left_idxs = [int(l) for l in widths[2]]
    right_idxs = [int(r) for r in widths[3]] 

    peaks_times = df["retentionTime"][peak_idxs].values
    peaks_y = df[val_col][peak_idxs].values
    left_times = df["retentionTime"][left_idxs].values
    right_times = df["retentionTime"][right_idxs].values
    peaks_width_y = widths[1]
    
    return pd.DataFrame.from_dict({"peaks_times": peaks_times, 
                                   "peaks_y": peaks_y,
                                   "left_times": left_times,
                                   "right_times": right_times,
                                   "peaks_width_y": peaks_width_y})


def process_gc_ms():

    """
    """

    pass