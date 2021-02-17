#!/usr/bin/env python3

import pandas as pd
import numpy as np

# TODO: Incorporate 16-bit roll-over values from DAQ!!!
# TODO: encode frame pre/post cam trig >> frame delta term
# (10) as a variable in constants.py and then import? 


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


def get_freq_from_datetimes(df):

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

    assert len(datetime_cols)==1, \
        "df has more than 1 column of datetime objs"

    return 1 / np.mean(df[datetime_cols[0]].diff()).total_seconds()


def get_precam_duration(daq):

    """
    Get the duration (secs) prior to cam trigger start-up.

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data

    Returns:
    --------
    Duration (secs) before cam trigger start-up (float)
    """

    df = daq.loc[daq["DAQ count"]==0]
    precam_start_times = df["datetime"]
    precam_timedelta = precam_start_times.iloc[-1] - precam_start_times.iloc[0]
    precam_duration = precam_timedelta.total_seconds()

    return precam_duration # secs


def get_postcam_duration(daq):

    """
    Get the duration (secs) after cam trigger ending.

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data

    Returns:
    --------
    Duration (secs) after cam trigger ending (float)
    """

    max_count = daq["DAQ count"].max()

    df = daq.loc[daq["DAQ count"]==max_count]
    postcam_end_times = df["datetime"]
    postcam_timedelta = postcam_end_times.iloc[-1] - postcam_end_times.iloc[0]
    postcam_duration = postcam_timedelta.total_seconds()

    return postcam_duration # secs


def get_cam_deltasecs_from_daq(daq):
    
    """
    Get the mean time delta in seconds, every time the 
    frame count signal from the cam trigger increments. 

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data

    Returns:
    --------
    Mean time delta in seconds (float)
    """

    assert daq["DAQ count"].iloc[0]==0, \
        "DAQ counts do not start from 0"
    
    # Get the datetime whenever the frame increments:
    df = daq.loc[daq["DAQ count"].diff() > 0]
    datetimes = df["datetime"].to_frame(name="datetime")

    mean_deltasecs = 1 / get_freq_from_datetimes(datetimes) 

    return mean_deltasecs # secs


def get_cam_deltasecs_from_imgs(fmf_times):

    """
    
    """

    pass


def get_daq_frame_skips(daq):

    """
    Get data from DAQ during which frame skips 
    happened.

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data

    Returns:
    --------
    A Pandas dataframe during which frames skipped.
    """

    skipped_frames = daq.loc[daq["DAQ count"].diff() > 1]

    return skipped_frames


def get_img_frame_skips(fmf_times):

    """
    
    """

    pass


def did_frames_skip(daq, fmf_times):

    """
    """

    if len(get_daq_frame_skips(daq)) == 0:
        # TODO
        pass
    else:
        return False

    if len(get_img_frame_skips(fmf_times)) == 0:
        # TODO
        pass
    else:
        return False


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
    if daq["datetime"].iloc[0] < motor["datetime"].iloc[0]:
        print("\u2714 DAQ start \u2192 motor start")
    else:
        print("\u274C DAQ start \u2192 motor start")
        return False

    # Motor starts before cam trigger?
    assert daq["DAQ count"].iloc[0]==0, \
        "DAQ counts do not start from 0"

    # Check that cam trigger started last by checking that frame counts
    # stayed at 0 on start-up for longer than expected:
    if get_precam_duration(daq) > 10 * get_cam_deltasecs_from_daq(daq): 
        print("\u2714 motor start \u2192 cam trigger start")
    else:
        print("\u274C motor start \u2192 cam trigger start")
        return False

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

    # Cam trigger finishes collecting before motor?   
    if get_postcam_duration(daq) > 10 * get_cam_deltasecs_from_daq(daq):
        print("\u2714 cam trigger end \u2192 motor end")
    else:
        print("\u274C cam trigger end \u2192 motor end")
        return False

    # Motor finishes collecting before DAQ?
    if daq["datetime"].iloc[-1] > motor["datetime"].iloc[-1]:
        print("\u2714 motor end \u2192 DAQ end")
    else:
        print("\u274C motor end \u2192 DAQ end")
        return False

    return True


def main():

    cam_hz = 100 # the set freq, which should be user inputted
    fail_msg = ""

    # Specify data (.csvs with "motor" and "daq" keywords) 
    # default to all data

    # Transform data to correct object types (convert datetime strs to datetime objs)

    # check that get_cam_deltasecs_from_daq() == get_cam_deltasecs_from_imgs()
    # and return all mismatches

    # Keep track of data the script has succesfully inspected!

    pass


if __name__ == "__main__":
    main()