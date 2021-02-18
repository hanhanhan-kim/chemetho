#!/usr/bin/env python3

from pathlib import Path
import math

import pandas as pd
import numpy as np
import motmot.FlyMovieFormat.FlyMovieFormat as FMF

# TODO: Incorporate 16-bit roll-over values from DAQ!!!
# TODO: encode frame pre/post cam trig >> frame dt (10x)
# as a variable in constants.py and then import? 


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


def are_dts_close(set_dt, real_dt):

    """
    Checks if 2 dts are within an order of magnitude 
    precision of each other. 

    Both dts must have the same units.

    Parameters:
    -----------
    set_dt (float): the set dt, 
        e.g. 1 / set framerate
    real_dt(float): the actual dt, 
        e.g. the diff between 2 frames 

    Returns:
    --------
    Boolean: True, if both dts are close.
    """

    abs_tol = set_dt / 10
    return math.isclose(set_dt, real_dt, 
                        abs_tol=abs_tol)


def get_cam_dts_from_daq(daq):
    
    """
    Get the dt in seconds, every time the 
    frame count signal from the cam trigger increments. 

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data

    Returns:
    --------
    A Pandas dataframe of dts (secs) 
    """

    assert daq["DAQ count"].iloc[0]==0, \
        "DAQ counts do not start from 0"
    
    # Get the datetime whenever the frame increments:
    df = daq.loc[daq["DAQ count"].diff() > 0]
    timedeltas = df["datetime"].diff()
    dts = timedeltas.dt.total_seconds() 

    # TODO: Rename column in dts to "dts (secs)"
    return dts # secs


def get_cam_dts_from_fmfs(fmfs):

    """
    Get the dt in seconds, every time the 
    frame increments, for each video in a list of .fmf 
    videos.

    Parameters:
    -----------
    fmfs: A list of related .fmf video objects

    Returns:
    --------
    A list of Pandas dataframes with dt (secs) 
    """

    all_elapsed_secs = [fmf.get_all_timestamps() 
                        for fmf in fmfs]
    all_dts = np.diff(all_elapsed_secs)

    # Convert list of np arrays into list of dfs, for consistency:
    all_dts = [pd.DataFrame(dts, columns=["dts (secs)"]) 
               for dts in all_dts]

    return all_dts


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


def get_img_frame_skips(fmfs, set_dt):

    """
    Get data from .fmf videos during which frame skips 
    happened. 

    Parameters:
    -----------
    fmfs: A list of related .fmf video objects. 
        They should all have the same set dt or
        framerate. 
    set_dt (float): the set dt between frames, 
        e.g. 1 / set framerate
    
    Returns:
    --------
    A list of Pandas dataframes during which frames skipped.
    """

    all_dts = get_cam_dts_from_fmfs(fmfs)

    # For each fmf, get the indices of all skipped frames:
    all_skipped_idxs = [[i for i,dt in enumerate(dts.to_numpy()) 
                         if not are_dts_close(set_dt, dt)] 
                        for dts in all_dts]
    # N.B. I don't directly know if frames were skipped, 
    # I have to estimate from dt dissimilarities. 

    # Turn into list of dataframes: 
    all_skipped_frames = [pd.DataFrame(skipped_idxs, columns=["frame no."]) 
                          for skipped_idxs in all_skipped_idxs]

    return all_skipped_frames


def did_frames_skip(daq, fmfs, set_dt):

    """
    Check that frames didn't skip, according to 
    both the DAQ and the .fmf videos. 

    Parameters:
    -----------
    daq: Pandas dataframe of LabJack DAQ data
    fmfs: A list of related .fmf video objects. 
        They should all have the same set dt or
        framerate. 
    set_dt (float): the set dt between frames, 
        e.g. 1 / set framerate

    Returns:
    --------
    Boolean: False if frames did not skip.
    """

    # If the DAQ didn't count any frame signal skips: 
    if len(get_daq_frame_skips(daq)) == 0:
        pass
    else:
        return True

    # If all .fmfs have no skipped frames:
    all_skipped_frames = get_img_frame_skips(fmfs, set_dt)
    if all([df.empty for df in all_skipped_frames]):
        pass
    else:
        return True
    
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
    Boolean: True if start-up is good.
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
    if get_precam_duration(daq) > 10 * np.mean(get_cam_dts_from_daq(daq)): 
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
    Boolean: True if ending is good.
    """

    # Cam trigger finishes collecting before motor?   
    if get_postcam_duration(daq) > 10 * np.mean(get_cam_dts_from_daq(daq)):
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
    set_dt = 1/cam_hz
    fail_msg = ""
    daq = "" # df ... pd.read_csv(path)
    motor = "" # df ... pd.read_csv(path)
    fmfs = "" # list of fmf objects

    # Specify data (.csvs with "motor" and "daq" keywords) 
    # default to all data
    # TODO: Need to figure out how to keep the groupings of related data files

    # Transform data to correct object types (convert datetime strs to datetime objs)

    if is_ending_good(daq, motor):
        pass
    else:
        print(fail_msg)
        return False

    if is_startup_good(daq, motor):
        pass
    else:
        print(fail_msg)
        return False

    if not did_frames_skip(daq, fmfs, set_dt):
        pass
    else:
        print(fail_msg)
        return False

    # check that get_cam_dts_from_daq() == get_cam_dts_from_imgs()
    # and return all mismatches

    # Keep track of data the script has succesfully inspected!
    # Won't be worth the work though, if compute is fast. 

    return True


if __name__ == "__main__":
    main()