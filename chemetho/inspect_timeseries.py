#!/usr/bin/env python3

import pandas as pd
import numpy as np


def detect_datetime_cols(df):
    
    """
    Detect columns with datetime objects. 

    Parameters:
    -----------
    df: A Pandas dataframe

    Returns:
    --------
    A list of datetime column names
    """

    datetime_cols = [col for col in df.select_dtypes(include="datetime").columns]
    return datetime_cols


def get_freq(df):

    """
    Get mean frequency from datetime object column.

    Parameters:
    -----------
    df: A Pandas dataframe

    Returns:
    --------
    The average frequency in Hz (float)
    """

    datetime_cols = detect_datetime_cols(df)

    assert len(datetime_cols)==1, "df has more than 1 column of datetime objs"

    return 1 / np.mean(df[datetime_cols].diff()).total_seconds()


def datetimes_start_at_0(df):

    datetime_cols = detect_datetime_cols(df)
    starts_at_0s = [df[col].iloc[0] == 0 for col in datetime_cols]
    assert all(starts_at_0s), "not all datetime cols start at 0" # TODO: print the datetime col that doesn't


def is_startup_good(daq, motor):
    
    """
    Checks that the order of acquisition events during 
    start-up is correct.

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data
    motor: Pandas dataframe of Autostep motors' positions

    Returns:
    --------
    boolean
    """

    assert daq.select_dtypes(include="datetime") is not None, \
        "`daq` df does not have any columns with datetime objs"
    assert motor.select_dtypes(include="datetime") is not None, \
        "`motor` df does not have any columns with datetime objs"

    # DAQ starts before motor?
    assert daq["datetime"].iloc[0] < motor["datetime"].iloc[0] == True, \
        "DAQ does NOT start collecting before motor stream"
    print("\u2714 DAQ start \u2192 motor start")

    # Motor starts before cam trigger?
    # TODO assert
    print("\u2714 motor start \u2192 cam trigger start")

    return True


def is_ending_good(daq, motor):
    
    """
    Checks that the order of acquisition events during 
    completion is correct.

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data
    motor: Pandas dataframe of Autostep motors' positions

    Returns:
    --------
    boolean
    """

    # Motor finishes collecting before DAQ?
    assert daq["datetime"].iloc[-1] > motor["datetime"].iloc[-1] == True, \
        "Motor does NOT finish collecting before DAQ"
    print("\u2714 motor end \u2192 DAQ end")

    # Cam trigger finishes collecting before motor?
    # TODO assert
    print("\u2714 cam trigger end \u2192 motor end")

    return True


def main():

    cam_hz = 100 # the set freq, which should be user inputted

    # Specify data (.csvs with "motor" and "daq" keywords) 
    # default to all data

    # Transform data to correct object types (convert to datetime objs)

    pass


if __name__ == "__main__":
    main()