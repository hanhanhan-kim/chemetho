#!/usr/bin/env python3

"""
Inspect timeseries data from NoEXIIT.

Checks that the order of acquisition events during 
both start-up and ending are as expected, and that 
no frames were skipped during video capture.

Assumes that data from the same experiment are found
in the same leaf subdirectory.  

Script outputs only print messages to terminal. 
"""

import argparse 
from pathlib import Path
from os.path import expanduser
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

    # Turn into list of dfs, for consistency:
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

    # Turn into list of dfs, for consistency:: 
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
        print("\u2714 DAQ did not detect any skipped frames")
    else:
        print("\u274C DAQ detected skipped frames")
        return True

    # If no .fmfs have skipped frames:
    all_skipped_frames = get_img_frame_skips(fmfs, set_dt)
    if all([df.empty for df in all_skipped_frames]):
        print("\u2714 timestamps from .fmfs suggest no skipped frames")
    else:
        print("\u274C timestamps from .fmfs detected skipped frames")
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
        print("\u2714 DAQ start\u2192 motor start")
    else:
        print("\u274C DAQ start\u2192 motor start")
        return False

    # Motor starts before cam trigger?
    assert daq["DAQ count"].iloc[0]==0, \
        "DAQ counts do not start from 0"

    # Check that cam trigger started last by checking that frame counts
    # stayed at 0 on start-up for longer than expected:
    if get_precam_duration(daq) > 10 * np.mean(get_cam_dts_from_daq(daq)): 
        print("\u2714 motor start\u2192 cam trigger start")
    else:
        print("\u274C motor start\u2192 cam trigger start")
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
        print("\u2714 cam trigger end\u2192 motor end")
    else:
        print("\u274C cam trigger end\u2192 motor end")
        return False

    # Motor finishes collecting before DAQ?
    if daq["datetime"].iloc[-1] > motor["datetime"].iloc[-1]:
        print("\u2714 motor end\u2192 DAQ end")
    else:
        print("\u274C motor end\u2192 DAQ end")
        return False

    return True


def main():

    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("datapath", help="Path to root data directory.")
    parser.add_argument("set_hz", help="Target framerate of cam trigger in Hz.")

    args = parser.parse_args()

    datapath = expanduser(args.datapath)
    set_dt = 1 / float(args.set_hz)

    assert Path(datapath).is_dir(), f"{datapath} is not a directory"

    fmf_paths = [str(path) for path in Path(datapath).rglob("*.fmf")] 
    csv_paths = [str(path) for path in Path(datapath).rglob("*.csv")] 
    paths = fmf_paths + csv_paths

    # Get leaf subdirectories: 
    # N.B. assumes unique expt is date -> animal -> trial
    splitted = ["/".join(path.split("/")[-4:-1]) for path in paths]
    leaves = set(splitted)

    # Sort into leaf:data pairs:
    expts = {}
    for path in paths:
        for leaf in leaves: 
            if leaf in path:
                if not leaf in expts.keys():
                    expts[leaf] = [path] # make key
                else:
                    expts[leaf].append(path)

    # Process and inspect all data:
    for expt, dataset in expts.items():        

        print(f"\nexpt: {expt}")
        print("-------------------------------")

        assert len([data for data in dataset if "daq" in data]) == 1, \
            "The no. of DAQ .csv outputs is not exactly 1"
        assert len([data for data in dataset if "motor" in data]) == 1, \
            "The no. of Autostep .csv outputs is not exactly 1"
        assert len([data for data in dataset if ".fmf" in data]) == 5, \
            "The no. of .fmf videos is not exactly 5"
        fail_msg = f"\u274C The experiment from {expt} failed the inspection"
        
        fmf_paths = []
        for data in dataset:

            if ".fmf" in data:
                fmf_paths.append(data)
            elif "daq" in data:
                daq = pd.read_csv(data)
            elif "motor" in data:
                motor = pd.read_csv(data)
            else:
                print(f"Unrecognized file type in experiment, {expt}")
                
        # Parse dataset into correct types:
        fmfs = [FMF.FlyMovie(path) for path in fmf_paths]
        motor["datetime"] = pd.to_datetime(motor["datetime"], 
                                        format="%Y-%m-%d %H:%M:%S.%f")
        daq["datetime"] = pd.to_datetime(daq["datetime"], 
                                        format="%Y-%m-%d %H:%M:%S.%f")
        
        # Finally, inspect the dataset:
        print(f"DAQ frequency: {get_freq_from_datetimes(daq):.2f} Hz")
        print(f"Autostep frequency: {get_freq_from_datetimes(motor):.2f} Hz")

        trig_dt = np.mean(get_cam_dts_from_daq(daq)) # from DAQ
        all_dts = get_cam_dts_from_fmfs(fmfs) # from fmfs
        mean_dts = [np.mean(dts).to_numpy() for dts in all_dts]

        if all([are_dts_close(mean_dt, trig_dt) for mean_dt in mean_dts]):
            print(f"Video frequencies: {1/trig_dt:.2f} Hz")
        else:
            print(fail_msg)
            break
    
        if is_startup_good(daq, motor):
            pass
        else:
            print(fail_msg)
            break

        if is_ending_good(daq, motor):
            pass
        else:
            print(fail_msg)
            break

        if not did_frames_skip(daq, fmfs, set_dt):
            pass
        else:
            # TODO: print all skipped frame instances here
            print(fail_msg) 
            break
        
    print("")


if __name__ == "__main__":
    main()