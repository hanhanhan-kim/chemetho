#!/usr/bin/env python3

import glob 
from pathlib import Path

import numpy as np
import scipy.interpolate as spi 
import pandas as pd

from .common import unconcat


def convert_noexiit_servo(df, servo_min, servo_max, servo_touch):
    
    """
    Converts the NoEXIIT robot's servo commands into real distances (mm). 
    
    Parameters:
    ------------
    df: A dataframe

    servo_min (fl): The minimum linear servo extension. Must be in mm.

    servo_max (fl): The maximum linear servo extension. Must be in mm. 

    servo_touch (fl): The linear servo extension length (mm) at which the stimulus 
        touches the insect on the ball. Will often be the same as 'servo_max'. 

    Returns:
    ---------
    A Pandas dataframe. 
    """
    
    assert "Servo output (degs)" in df,  "The column 'Servo output (degs)' is not in the dataframe, 'df'."

    # Generate function to map servo parameters:
    f_servo = spi.interp1d(np.linspace(0,180), np.linspace(servo_min, servo_max))

    # Map:
    df["Servo output (mm)"] = f_servo(df["Servo output (degs)"])
    
    # Convert to distance from stim: 
    df["dist_from_stim_mm"] = servo_touch - df["Servo output (mm)"]
        
    return df


def make_noexiit_trajectory(df):

    """
    Compute the NoEXIIT robot's trajectory, relative to the tethered animal.
    
    Assumes that the reference direction of the robot angle is parallel and 
    co-linear to the tethered animal in real space.  

    Parameters:
    -----------
    df: A Pandas dataframe. 
    
    Return:
    -------
    A dataframe with the NoEXIIT robot's X and Y coordinates. 
    """

    assert("X_mm" in df), "The dataframe must have a column called 'X_mm'"
    assert("Y_mm" in df), "The dataframe must have a column called 'Y_mm'"
    assert("dist_from_stim_mm" in df), "The dataframe must have a column called 'dist_from_stim_mm'"

    def compute_X_mm(row):
        X_mm = row["X_mm"] + (row["dist_from_stim_mm"] * np.cos(np.deg2rad(row["Stepper output (degs)"])))
        return X_mm

    def compute_Y_mm(row):
        Y_mm = row["Y_mm"] + (row["dist_from_stim_mm"] * np.sin(np.deg2rad(row["Stepper output (degs)"])))
        return Y_mm

    df['other_X_mm'] = df.apply(compute_X_mm, axis=1)
    df['other_Y_mm'] = df.apply(compute_Y_mm, axis=1) 
        
    return df