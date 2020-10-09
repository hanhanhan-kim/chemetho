import pandas as pd
import numpy as np

def get_relative_intensity_df(df):
    
    """
    Converts a pyteeomics.mzxml -> dataframe of Shimadzu GC-MS data into
    a dataframe of relative intensities, where the raw count data is 
    normalized by the total number of counts. 
    
    Parameters:
    -----------
    df: A dataframe generated from pyteomics.mzxml.read()
    
    Returns:
    --------
    A dataframe of mass/charge, retention time, and relative intensity
    """
    
    rel_ints_dfs = []
    
    grouped = df.groupby(['lowMz'])
    for _, group in enumerate(grouped):
        
        # group[0] holds the m/z:
        low_mz = f"{group[0]}"
        
        # group[1] holds the actual dataframe:
        ret_times = group[1]['retentionTime']
        
        raw = [np.sum(val) for val in group[1]['intensity array'].values]
        denom = np.sum(group[1]['intensity array'].values)

        rel_ints = raw / denom
        
        rel_ints_df = pd.DataFrame.from_dict({"mass/charge": low_mz,
                                              "retention time (mins)": ret_times, 
                                              "relative intensity": rel_ints}
                                            ).reset_index(drop=True)
        
        rel_ints_dfs.append(rel_ints_df)
    
    return(pd.concat(rel_ints_dfs))