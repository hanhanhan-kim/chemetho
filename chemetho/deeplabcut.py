import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import glob

import bokeh.plotting
import bokeh.io
from bokeh.layouts import column
from bokeh.models import Legend
from bokeh.models.ranges import Range1d


def plot_TrainingStats(TrainingStats, alpha=0.6, size=15, line_width=4):
    '''
    Creates a Bokeh plotting object that plots the training and test errors (without p-cutoff) of the .csv that DeepLabCut generates after evaluating the DLC network.
    
    Parameters:
    TrainingStats: The dataframe generated from the default .csv file DeepLabCut generates after evaluating the DLC network.
    alpha: plotting aesthetic
    size: plotting aesthetic
    line_width: plotting aesthetic
    '''
    
    p = bokeh.plotting.figure(height=400,
                              width=600,
                              x_axis_label='Iteration Number',
                              y_axis_label='Error as Euclidean distance (px)')

    # Training error(px)
    p.line(source=TrainingStats,
           x='Training iterations:',
           y=' Train error(px)',
           color='red',
           legend='Train',
           line_width=line_width,
           alpha=alpha)
    p.circle(source=TrainingStats,
             x='Training iterations:',
             y=' Train error(px)', 
             color='red', 
             legend='Train',
             size=size)

    # Test error(px)
    p.line(source=TrainingStats,
           x='Training iterations:',
           y=' Test error(px)',
           color='blue',
           legend='Test',
           line_width=line_width,
           alpha=alpha)
    p.circle(source=TrainingStats,
             x='Training iterations:',
             y=' Test error(px)', 
             color='blue', 
             legend='Test',
             size=size)

    # Disable sci not
    p.below[0].formatter.use_scientific = False

    # Adjust spacing for legend
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "9pt"
    p.y_range = bokeh.models.ranges.Range1d(0, max(TrainingStats[' Test error(px)'])+2)

    return p


#
def compute_angle (p_1_x, p_1_y, p_2_x, p_2_y, p_3_x, p_3_y):
    '''
    Computes the angle between 3 coordinate points. Assumes the first two arguments specify the (x,y) coords of the vertex.
    
    Parameters:
    p_1_x: x-coord of the vertex of the 3 points
    p_1_y: y-coord of the vertex of the 3 points
    p_2_x: x-coord of an arm end of the 3 points
    p_2_y: y-coord of an arm end of the 3 points
    p_3_x: x-coord of the other arm end of the 3 points
    p_3_y: y-coord of the other arm end of the 3 points
    '''
    
    # Compute distance between each pair of points:
    p_12 = np.sqrt((p_1_x - p_2_x)**2 + (p_1_y - p_2_y)**2)
    p_13 = np.sqrt((p_1_x - p_3_x)**2 + (p_1_y - p_3_y)**2)
    p_23 = np.sqrt((p_2_x - p_3_x)**2 + (p_2_y - p_3_y)**2)
    
    # Use Law of Cosines to compute angle:
    angle = np.arccos((p_12**2 + p_13**2 - p_23**2)/(2 * p_12 * p_13))
    
    return (np.rad2deg(angle))


#
def preprocess_AnalyzedBeetleonballVid(names, framerate):
    '''
    Takes a list of .csv files generated by deeplabcut.analyze_videos and returns a list of pre-processed dataframes.
    Each pre-processed dataframe is no longer multi-indexed, and elapsed recording times are provided in frames, seconds, and minutes.
    This function assumes that the analyzed video is of a beetle on an air-supported ball, and that exactly 8 body parts are labelled.
    
    Parameters:
    names: a list of default .csv files generated by deeplabcut.analyze_videos
    framerate: the framerate of acquistion of the analyzed videos
    '''
    
    dfs = []
    
    for name in names:
        # Read in .csv file:
        df = pd.read_csv(name, header=[0,1,2])
        
        # Reformat the dataframes:
        df.columns = df.columns.get_level_values(1) + ['_' for col in df.columns] + df.columns.get_level_values(2)
        
        # Rename the first column of the dataframes:
        df = df.rename(index=str, columns={'bodyparts_coords': 'frame'})
        
        # Compute elapsed time
        df['secs_elapsed'] = df['frame']/framerate
        df['mins_elapsed'] = df['secs_elapsed']/60
        
        dfs.append(df)
        
    return dfs


#
def plot_side_XvsYpixels(proc_df, palette=bokeh.palettes.Spectral8, alpha=0.05, ymax=540, ymin=0):
    '''
    
    '''
    
    # Generate two lists; of column headings that end with _x and of column headings that end with _y:
    endswith_x = [col for col in proc_df if col.endswith('_x')]
    endswith_y = [col for col in proc_df if col.endswith('_y')]

    # Plot:
    p = bokeh.plotting.figure(height=500,
                          width=500,
                          x_axis_label='X (pixels)',
                          y_axis_label='Y (pixels)')

    for idx, bodypart in enumerate(endswith_x):
        p.circle(x=proc_df[endswith_x[idx]],
                 y=proc_df[endswith_y[idx]],
                 color = palette[idx], 
                 legend=endswith_x[idx].split('_x')[0],
                 alpha=alpha)

    # Flip the y-axis:
    p.y_range = Range1d(ymax, ymin)
    
    # Adjust legend:
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "9pt"
    p.legend.background_fill_alpha = 0.0
    p.legend.visible = False # CHANGE IF DESIRED
    
    bokeh.io.show(p)


#
def plot_top_XvsYpixels(proc_df, palette=bokeh.palettes.Spectral8, alpha=0.05, xmax=720, xmin=0):
    '''
    
    '''
    
    # Generate two lists; of column headings that end with _x and of column headings that end with _y:
    endswith_x = [col for col in proc_df if col.endswith('_x')]
    endswith_y = [col for col in proc_df if col.endswith('_y')]

    # Plot:
    p = bokeh.plotting.figure(height=500,
                          width=500,
                          x_axis_label='X (pixels)',
                          y_axis_label='Y (pixels)')

    for idx, bodypart in enumerate(endswith_x):
        p.circle(x=proc_df[endswith_x[idx]],
                 y=proc_df[endswith_y[idx]],
                 color = palette[idx], 
                 legend=endswith_x[idx].split('_x')[0],
                 alpha=alpha)

    # Flip the x-axis:
    p.x_range = Range1d(xmax, xmin)
    
    # Adjust legend:
    p.legend.location = "bottom_right"
    p.legend.orientation = "horizontal"
    p.legend.label_text_font_size = "9pt"
    p.legend.background_fill_alpha = 0.0
    p.legend.visible = False # CHANGE IF DESIRED
    
    bokeh.io.show(p)