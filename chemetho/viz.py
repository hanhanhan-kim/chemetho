from os.path import join

import numpy as np
import scipy.interpolate as spi
from pylab import matplotlib
from bokeh.io import output_file, export_png, export_svgs, show
from bokeh.transform import linear_cmap
from bokeh.plotting import figure
from bokeh.models import ColorBar, ColumnDataSource, Span, BoxAnnotation
from bokeh.layouts import gridplot
import bokeh.palettes
import colorcet as cc
import cmocean
import iqplot

from .constants import themes, banned_substrings
from .common import ban_columns_with_substrings, aggregate_trace
from .fourier import fft 


def cmap2hexlist(cmap):
    
    """
    Converts a matplotlib.colors.LinearSegementedColormap object to 
    a list of hex colours. Useful for converting attribute colour maps 
    of cmocean.cm for use with bokeh. 
    
    Parameters:
    -----------
    cmap: a matplotlib.colors.LinearSegmentedColormap object. 
        Attributes of cmocean.cm are these objects and are colour maps. 
        E.g. cmocean.cm.haline
        
    Returns:
    --------
    A list of hexadecimal colour codes. 
    """
    
    hexcolours = []
    for i in range(cmap.N):
        # Returns rgba; take only first 3 to get rgb:
        rgb = cmap(i)[:3] 
        hexcolours.append(matplotlib.colors.rgb2hex(rgb))
    
    return hexcolours


def get_all_cmocean_colours():
    
    """
    Returns a dictionary of the cmocean colour maps.
    Each key in the dictionary is the name of the colourmap and each value 
    in the dictionary is a list of the colourmap's colours in hexadecimal. 
    """
    
    cmocean_attrs = dir(cmocean.cm)

    hexcolourmaps_dict = {}
    for cmocean_attr in cmocean_attrs:

        attr_instance = getattr(cmocean.cm, cmocean_attr)
        attr_type = type(attr_instance)

        if attr_type is matplotlib.colors.LinearSegmentedColormap:
            
            # Convert val to hexlist and add key-val pair:
            hexcolourmaps_dict[cmocean_attr] = cmap2hexlist(attr_instance)
    
    return hexcolourmaps_dict


cm = get_all_cmocean_colours()


def load_plot_theme(p, theme=None, has_legend=False):
    
    """
    Load theme colours into a Bokeh plotting object. 

    Parameters:
    -----------
    p: A Bokeh plotting object
    
    Returns:
    --------
    `p` with coloured attributes. 
    """
    
    assert (theme in themes or theme is None), \
        f"{theme} is neither None nor a valid key in `themes`"

    if theme is not None:

        theme_colours = themes[theme]
            
        dark_hue = theme_colours["dark_hue"]
        light_hue = theme_colours["light_hue"]

        p.border_fill_color = light_hue
        p.xgrid.grid_line_color = dark_hue
        p.ygrid.grid_line_color = dark_hue
        p.background_fill_color = light_hue 

        if has_legend == True:
            p.legend.background_fill_color = light_hue
        else:
            pass

        return p
    
    else:
        pass


def plot_power_spectra(df, val_cols, time_col, 
                        is_evenly_sampled=False, window=np.hanning, pad=1, 
                        cutoff_freq=None, 
                        val_labels=None, time_label=None,
                        theme=None,
                        save_path_to=None, show_plots=True):  

    """
    Perform a Fourier transform on FicTrac data for each ID. Plot 
    power spectrum for each value in `val_cols`. 
    Accepts one column from `df`, rather than a list of columns.

    Parameters:
    ------------
    df (DataFrame): Dataframe of FicTrac data generated from parse_dats()

    val_cols (list): List of column names from `df` to be Fourier-transformed.  

    time_col (str): Column name in `df` that specifies time in SECONDS. 

    is_evenly_sampled (bool): If False, will interpolate even sampling. 

    cutoff_freq (float): x-intercept value for plotting a vertical line. 
        To be used to visualize a candidate cut-off frequency. Default is None.
    
    val_labels (list): List of labels for the plot's y-axis. If None, will 
        be a formatted version of val_cols. Default is None.

    time_label (str): Label for the plot's time-axis.

    theme (str or None): A pre-defined colour theme for plotting. If None,
        does not apply a theme. Default is None.

    save_path_to (str): Absolute path to which to save the plots as .png files. 
        If None, will not save the plots. Default is None. 

    show_plots (bool): If True, will show plots, but will not output Bokeh plotting 
        objects. If False, will not show plots, but will output Bokeh plotting objects. 
        If both save and show_plots are True, .html plots will be generated, in addition 
        to the .png plots. Default is True.

    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """
    
    if not ('sec' in time_col or '(s)' in time_col):
        safe_secs = input(f"The substrings 'sec' or '(s)' was not detected in the 'time_col' variable, {time_col}. The units of the values in {time_col} MUST be in seconds. If the units are in seconds, please input 'y'. Otherwise input anything else to exit.")
        while True:
            if safe_secs.lower() == "y":
                break
            else:
                exit("Re-run this function with a 'time_col' whose units are secs.")

    assert (time_col in df), f"The column, {time_col}, is not in the input dataframe."
    
    time = list(df[str(time_col)])

    # Format axes labels:
    if time_label == None:
        time_label = time_col.replace("_", " ")
    if val_labels == None:
        val_labels = [val_col.replace("_", " ") for val_col in val_cols]


    def load_aesthetics(freq, amp):

        """
        Load aesthetics for plotting power spectra. 
        """

        title = "Frequency domain"
        
        # Frequency domain:
        p1 = figure(
                title=title,
                width=1000,
                height=500,
                y_axis_label="power"
        )
        p1.line(
            x=freq,
            y=amp,
            color="darkgray"
        )
        p2 = figure(
                width=1000,
                height=500,
                x_axis_label="frequency (Hz)",
                y_axis_label="log power",
                y_axis_type="log"
        )
        p2.line(
            x=freq,
            y=amp,
            color="darkgray"
        )

        return p1, p2


    plots = []
    for i, val_col, in enumerate(val_cols):

        assert (len(df[time_col] == len(df[val_col]))), \
            "time and val are different lengths! They must be the same."
        assert (val_col in df), \
            f"The column, {val_col}, is not in the input dataframe."
        
        val = list(df[str(val_col)])

        # Fourier-transform:
        f = spi.interp1d(time, val)

        if is_evenly_sampled == False:
            time_interp = np.linspace(time[1], time[-1], len(time))
            val_interp = f(time_interp)
        else:
            time_interp = time
            val_interp = val

        amp, _, freq = fft( val_interp, 
                            time_interp, 
                            pad=1, 
                            window=window, 
                            post=True)
        

        # Plot:
        p1, p2 = load_aesthetics(freq, amp)

        p1.title.text = f"power spectrum of {val_labels[i]}"
        p1.title.text_font_size = "16pt"
        p1.yaxis.axis_label_text_font_size = "12pt"
        load_plot_theme(p1, theme=theme)

        p2.yaxis.axis_label_text_font_size = "12pt"
        p2.xaxis.axis_label_text_font_size = "12pt"
        load_plot_theme(p2, theme=theme)

        if cutoff_freq is not None:
            float(cutoff_freq)
            cutoff_line = Span(location=cutoff_freq, 
                            dimension="height", 
                            line_color="#775a42",
                            line_dash="dashed",
                            line_width=2)
            p1.add_layout(cutoff_line)
            p2.add_layout(cutoff_line)

        p = gridplot([p1, p2], ncols=1)

        # Output:
        if save_path_to is not None:
            filename = save_path_to + f"fictrac_freqs"

            # Bokeh does not atm support gridplot svg exports
            export_png(p, filename = filename + ".png")
            output_file(filename = filename + ".html", 
                        title=f"fictrac_freqs")

        if show_plots == True:
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def plot_filtered(df, val_cols, time_col, 
                  order, cutoff_freq,
                  val_labels=None, time_label=None,
                  view_perc=100, 
                  theme=None,
                  save_path_to=None, show_plots=True):

    """
    Apply a low-pass Butterworth filter on offline FicTrac data. 
    Plot filtered vs. non-filtered data. 
    Purpose is to assess filter parameters on data. 

    Parameters:
    -----------
    df (DataFrame): Filtered dataframe of FicTrac data generated from parse_dats(). 
        Should have columns with the "filtered_" prefix. 

    val_cols (list): List of column names from `df` to be Fourier-transformed. 

    time_col (str): Column name from `df` that specifies time. 

    order (int): Order of the filter.

    cutoff_freq (float): The cutoff frequency for the filter in Hz.

    val_labels (list): List of labels for the plot's y-axis. If None, will 
        be a formatted version of cmap_cols. Default is None.

    time_label (str): Label for the plot's time-axis. 

    view_perc (float): Specifies how much of the data to plot as an initial 
        percentage. Useful for assessing the effectieness of the filter over longer 
        timecourses. Default is set to 1, i.e. plot the data over the entire 
        timecourse. Must be a value between 0 and 1.

    theme (str or None): A pre-defined colour theme for plotting. If None,
        does not apply a theme. Default is None.

    save_path_to (str): Absolute path to which to save the plots as .png and .svg files. 
        If None, will not save the plots. Default is None. 

    show_plots (bool): If True, will show plots, but will not output Bokeh plotting 
        objects. If False, will not show plots, but will output Bokeh plotting objects. 
        If both save and show_plots are True, .html plots will be generated, in addition 
        to the .png plots. Default is True.

    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """
    
    assert (0 <= view_perc <= 100), \
        f"The view percentage, {view_perc}, must be between 0 and 100."
    assert (len(df.filter(regex="^filtered_").columns) > 0), \
        "At least one column in the dataframe must begin with 'filtered_'"
    
    # Format axes labels:
    if time_label == None:
        time_label = time_col.replace("_", " ")
    if val_labels == None:
        val_labels = [val_col.replace("_", " ") for val_col in val_cols]
    
    # View the first _% of the data:
    domain = int(view_perc/100 * len(df[time_col])) 

    plots = []
    for i, val_col in enumerate(val_cols):
        assert (len(df[time_col] == len(df[val_col]))), \
            "time and vals are different lengths! They must be the same."
        assert (time_col in df), \
            f"The column, {time_col}, is not in the input dataframe."
        assert (val_col in df), \
            f"The column, {val_col}, is not in the input dataframe."
        assert ("filtered_" in val_col), \
            f"The column, {val_col}, does not begin with the 'filtered_' prefix." 
        
        # Plot:
        p = figure(
            width=1600,
            height=500,
            x_axis_label=time_label,
            y_axis_label=val_labels[i] 
        )
        p.line(
            x=df[time_col][:domain],
            y=df[val_col.replace("filtered_","")][:domain],
            color=bokeh.palettes.brewer["Paired"][3][0],
            legend_label="raw"
        )
        p.line(
            x=df[time_col][:domain],
            y=df[val_col][:domain],
            color=bokeh.palettes.brewer["Paired"][3][1],
            legend_label="filtered"
        )
        p.title.text = f"first {view_perc}% with butterworth filter: cutoff = {cutoff_freq} Hz, order = {order}"
        p.title.text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.axis_label_text_font_size = "12pt"
        load_plot_theme(p, theme=theme, has_legend=True) 

        # Output:
        if save_path_to is not None:
            filename = join(save_path_to, val_col)
            p.output_backend = "svg"
            export_svgs(p, filename=filename + ".svg")
            export_png(p, filename=filename + ".png")
            output_file(filename=filename + ".html", 
                        title=filename)
            
        if show_plots == True:
            # In case this script is run in Jupyter, change output_backend 
            # back to "canvas" for faster performance:
            p.output_backend = "canvas"
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def plot_trajectory(df, cmap_cols, low=0, high_percentile=95, respective=False, 
                    cmap_labels=None,
                    order=2, cutoff_freq=4, 
                    palette=cm["thermal"], size=2.5, alpha=0.3, 
                    theme=None,
                    show_start=False, 
                    save_path_to=None, show_plots=True):
    
    """
    Plot XY coordinates of the individual with a linear colourmap for a each element 
    in `cmap_cols`. 
    
    Parameters:
    -----------
    df (DataFrame): Dataframe of FicTrac data generated from parse_dats()

    low (float): The minimum value of the colour map range. Any value below the set 
        'low' value will be 'clamped', i.e. will appear as the same colour as 
        the 'low' value. The 'low' value must be 0 or greater. Default value 
        is 0.

    high_percentile (float): The max of the colour map range, as a percentile of the 
        'cmap_col' variable's distribution. Any value above the 'high_percentile'
        will be clamped, i.e. will appear as the same colour as the 
        'high_percentile' value. E.g. if set to 95, all values below the 95th 
        percentile will be mapped to the colour map, and all values above the
        95th percentile will be clamped. 

    respective (bool): If True, will re-scale colourmap for each individual to 
        their respective 'high_percentile' cut-off value. If False, will use
        the 'high_percentile' value computed from the population, i.e. from `df`. 
        Default is False. 

    cmap_cols (list): List of column names from `df` to be colour-mapped. 

    cmap_labels (list): List of labels for the plots' colourbars. If None, will 
        be a formatted version of cmap_cols. Default is None.

    order (int): Order of the filter.

    cutoff_freq (float): The cutoff frequency for the filter in Hz.

    palette (list): A list of hexadecimal colours to be used for the colour map.

    size (float): The size of each datapoint specifying XY location. 

    alpha(float): The transparency of each datapoint specifying XY location.
        Must be between 0 and 1.

    theme (str or None): A pre-defined colour theme for plotting. If None,
        does not apply a theme. Default is None.

    show_start (bool): If True, will plot a marking to explicitly denote the start 
        site. Default is False. 
    
    save_path_to (str): Absolute path to which to save the plots as .png and .svg files. 
        If None, will not save the plots. Default is None. 

    show_plots (bool): If True, will show plots, but will not output Bokeh plotting 
        objects. If False, will not show plots, but will output Bokeh plotting objects. 
        If both save and show_plots are True, .html plots will be generated, in addition 
        to the .png plots. Default is True.

    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """

    assert (low >= 0), "The low end of the colour map range must be non-negative"
    assert ("X_mm" in df), "The column, 'X_mm', is not in the input dataframe."
    assert ("Y_mm" in df), "The column, 'Y_mm', is not in the input dataframe."
  
    # Format axes labels:
    if cmap_labels == None:
        cmap_labels = [cmap_col.replace("_", " ") for cmap_col in cmap_cols]

    plots = []
    for i, cmap_col in enumerate(cmap_cols):
        
        assert (cmap_col in df), \
            f"The column, {cmap_col}, is not in the input dataframe."
        assert (len(df["X_mm"] == len(df["Y_mm"]))), \
            "X_mm and Y_mm are different lengths! They must be the same."

        if respective == False:
            # Normalize colourmap range to population:
            high = np.percentile(df[cmap_col], high_percentile)
        elif respective == True:
            # Individual animal sets its own colourmap range:
            high = np.percentile(df[cmap_col], high_percentile)
        
        source = ColumnDataSource(df)

        mapper = linear_cmap(field_name=cmap_col, 
                             palette=palette, 
                             low=low, 
                             high=high)
        
        p = figure(width=800,
                    height=800,
                    x_axis_label="X (mm)",
                    y_axis_label="Y (mm)")
        
        p.circle(source=source,
                 x="X_mm",
                 y="Y_mm",
                 color=mapper,
                 size=size,
                 alpha=alpha)
        
        if show_start == True:
            # Other options include .cross, .circle_x, and .hex:
            p.circle(x=df["X_mm"][0], 
                     y=df["Y_mm"][0], 
                     size=12,
                     color="darkgray",
                     fill_alpha=0.5)

        # TODO: also change colorbar labels so max has =< symbol
        # TODO: Change background colour of colour bar, according to theme
        color_bar = ColorBar(color_mapper=mapper['transform'], 
                             title=cmap_labels[i],
                             title_text_font_size="7pt",
                             width=10,
                             background_fill_color="#f8f5f2",
                             location=(0,0))

        p.add_layout(color_bar, "right")
        p.title.text_font_size = "14pt"
        p.xaxis.axis_label_text_font_size = '10pt'
        p.yaxis.axis_label_text_font_size = '10pt'
        load_plot_theme(p, theme=theme)

        # Output:
        if save_path_to is not None:
            filename = save_path_to + f"fictrac_XY_{cmap_col}"
            p.output_backend = "svg"
            export_svgs(p, filename=filename + ".svg")
            export_png(p, filename=filename + ".png")
            output_file(filename=filename + ".html", 
                        title=filename)
            
        if show_plots == True:
            # In case this script is run in Jupyter, change output_backend 
            # back to "canvas" for faster performance:
            p.output_backend = "canvas"
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def plot_trajectories(df, cmap_cols, low=0, high_percentile=95, respective=False, 
                      cmap_labels=None,
                      order=2, cutoff_freq=4, 
                      palette=cm["thermal"], 
                      other_palette=cm["ice"], 
                      size=2.5, alpha=0.3, theme=None,
                      show_start=False, 
                      save_path_to=None, show_plots=True):

    """
    Plot XY coordinates of two agents. 
    Plots a linear colourmap for a each element in `cmap_cols`, for both agents. 

    Parameters:
    ------------
    The parameters of this function are identical to `plot_trajectory`, but with the
    following addition:

    other_palette (list): A list of hexadecimal colour to be used for the other agent's colour map.
    
    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """

    assert ("other_X_mm" in df), "The column, 'other_X_mm' is not in the input dataframe"
    assert ("other_Y_mm" in df), "The column, 'other_Y_mm' is not in the input dataframe"

    # Format axes labels:
    if cmap_labels == None:
        cmap_labels = [cmap_col.replace("_", " ") for cmap_col in cmap_cols]

    plots = plot_trajectory(df, cmap_cols, low=low, high_percentile=high_percentile, respective=False,
                            cmap_labels=cmap_labels, 
                            order=order, cutoff_freq=cutoff_freq, 
                            palette=palette, size=size, alpha=alpha,
                            theme=theme,
                            show_start=show_start,
                            save_path_to=save_path_to, show_plots=False) 

    source = ColumnDataSource(df)   

    for i,(cmap_col, p) in enumerate(zip(cmap_cols, plots)):

        if respective is False:
            high = np.percentile(df[cmap_col], high_percentile)

        # Plot other agent:
        other_mapper = linear_cmap(field_name=cmap_col, 
                        palette=other_palette, 
                        low=low, 
                        high=high)

        # TODO: also change colorbar labels so max has =< symbol
        # TODO: Change background colour of colour bar, according to theme
        other_color_bar = ColorBar(color_mapper=other_mapper['transform'], 
                                   title="robot " + cmap_labels[i],
                                   title_text_font_size="7pt",
                                   background_fill_color="#f8f5f2",
                                   width=10,
                                   location=(0,0))

        p.add_layout(other_color_bar, "right")

        p.circle(source=source,
                 x="other_X_mm", 
                 y="other_Y_mm", 
                 color=other_mapper, 
                 size=size, 
                 alpha=alpha)

        # Output:
        if save_path_to is not None:
            filename = save_path_to + f"fictrac_XY_{cmap_col}"
            p.output_backend = "svg"
            export_svgs(p, filename=filename + ".svg")
            export_png(p, filename=filename + ".png")
            output_file(filename=filename + ".html", 
                        title=filename)
            
        if show_plots == True:
            # In case this script is run in Jupyter, change output_backend 
            # back to "canvas" for faster performance:
            p.output_backend = "canvas"
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def plot_histograms(df, cols=None, labels=None, 
                    cutoff_percentile=95,
                    theme=None,
                    save_path_to=None, show_plots=True): 

    """
    Generate histograms for multiple variables. Originally written for FicTrac data.

    Parameters:
    -----------
    df (DataFrame): Dataframe

    cols (list): List of strings specifying column names in `df`. If None, 
        uses default columns that specify real-world and 'lab' kinematic measurements. 
        See `banned_substrings`. 
        Otherwise, will use both input arguments AND the default columns. Default is None.

    labels (list): List of strings specifying the labels for the histograms' x-axes.
        Its order must correspond to 'cols'. 

    cutoff_percentile (float): Specifies the percentile of the AGGREGATE population data. 
        Plots a line at this value. Default is 95th percentile. 

    theme (str or None): A pre-defined colour theme for plotting. If None,
        does not apply a theme. Default is None.
        
    save_path_to (str): Absolute path to which to save the plots as .png and .svg files. 
        If None, will not save the plots. Default is None. 

    show_plots (bool): If True, will show plots, but will not output Bokeh plotting 
        objects. If False, will not show plots, but will output Bokeh plotting objects. 
        If both save and show_plots are True, .html plots will be generated, in addition 
        to the .png plots. Default is True.

    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """

    assert ("ID" in df), "The column 'ID' is not in in the input dataframe."
    
    ok_cols = ban_columns_with_substrings(df)

    if cols == None:
        cols = ok_cols
    else:
        cols = ok_cols + cols 
        for col in cols:
            assert (col in df), f"The column, {col}, is not in the input dataframe."
    
    if labels == None:
        labels = [col.replace("_", " ") for col in cols] 

    plots = []
    for i, col in enumerate(cols):
        p = iqplot.histogram(data=df,
                             cats=['ID'],
                             val=col,
                             density=True,
                             width=1000,
                             height=500)
        
        cutoff_line = Span(location=np.percentile(df[col], cutoff_percentile), 
                           dimension="height", 
                           # line_color="#e41a1c",
                           line_color = "#775a42",
                           line_dash="dashed",
                           line_width=2)
        
        p.legend.location = "top_right"
        p.legend.title = "ID"
        load_plot_theme(p, theme=theme, has_legend=True)
        p.title.text = f" with aggregate {cutoff_percentile}% mark"
        p.xaxis.axis_label = labels[i]
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.add_layout(cutoff_line)
            
        # Output:
        if save_path_to is not None:
            filename = join(save_path_to, f"fictrac_histogram_by_ID_{col}")
            p.output_backend = "svg"
            export_svgs(p, filename=filename + ".svg")
            export_png(p, filename=filename + ".png")
            output_file(filename=filename + ".html", 
                        title=filename)
        
        if show_plots == True:
            # In case this script is run in Jupyter, change output_backend 
            # back to "canvas" for faster performance:
            p.output_backend = "canvas"
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def plot_ecdfs(df, cols=None, labels=None, 
               cutoff_percentile=95, 
               theme=None,
               save_path_to=None, show_plots=True):

    """
    Generate ECDFs for multiple variables. Originally written for FicTrac data.

    Parameters:
    -----------
    df (DataFrame): Dataframe

    cols (list): List of strings specifying column names in 'df'. If None, 
        uses default columns that specify real-world and 'lab' kinematic measurements.
        See `banned_substrings`
        Otherwise, will use both input arguments AND the default columns. Default is None.

    labels (list): List of strings specifying the labels for the ECDFs' x-axes.
        Its order must correspond to 'cols'. 

    cutoff_percentile (float): Specifies the percentile of the AGGREGATE population data. 
        Plots a line at this value. Default is 95th percentile. 

    theme (str or None): A pre-defined colour theme for plotting. If None,
        does not apply a theme. Default is None.
    
    save_path_to (str): Absolute path to which to save the plots as .png and .svg files. 
        If None, will not save the plots. Default is None. 

    show_plots (bool): If True, will show plots, but will not output Bokeh plotting 
        objects. If False, will not show plots, but will output Bokeh plotting objects. 
        If both save and show_plots are True, .html plots will be generated, in addition 
        to the .png plots. Default is True.

    Returns:
    ---------
    if show_plots is True: will show plots instead of outputting bokeh.plotting.figure object.
    if show_plots is False: will output bokeh.plotting.figure objects, instead of showing plots.
    if save_path_to is not None: will save .png plots to specified path. 
    if save_path_to is None: will not save plots.
    """

    assert ("ID" in df), "The column 'ID' is not in in the input dataframe."
    
    ok_cols = ban_columns_with_substrings(df)

    if cols == None:
        cols = ok_cols
    else:
        cols = ok_cols + cols 
        for col in cols:
            assert (col in df), f"The column, {col}, is not in the input dataframe."
    
    if labels == None:
        labels = [col.replace("_", " ") for col in cols] 

    plots = []
    for i, col in enumerate(cols):
        p = iqplot.ecdf(data=df,
                        cats=["ID"],
                        val=col,
                        kind="colored",
                        width=1000,
                        height=500)
        
        cutoff_line = Span(location=np.percentile(df[col], cutoff_percentile), 
                           dimension="height", 
                           line_color="#775a42",
                           line_dash="dashed",
                           line_width=2)
        
        p.legend.location = 'top_right'
        p.legend.title = "ID"
        load_plot_theme(p, theme=theme, has_legend=True)
        p.title.text = f" with aggregate {cutoff_percentile}% mark"
        p.xaxis.axis_label = labels[i]
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.add_layout(cutoff_line)
        
        # Output:
        if save_path_to is not None:
            filename = save_path_to + f"fictrac_ecdfs_{col}"
            
            p.output_backend = "svg"
            export_svgs(p, filename=filename + ".svg")
            export_png(p, filename=filename + ".png")
            output_file(filename=filename + ".html", 
                        title=filename)
            
        if show_plots == True:
            # In case this script is run in Jupyter, change output_backend 
            # back to "canvas" for faster performance:
            p.output_backend = "canvas"
            show(p)

        # In case show_plots is False:
        plots.append(p)

    if show_plots == False:
        return plots


def add_stimulus_annotation (p, style,
                             start, end, top, bottom, 
                             alpha, level="underlay"):

    """
    
    Add a stimulus annotation to a Bokeh plotting object. 

    Parameters:
    -----------
    p: A Bokeh plotting object
    style (str): One of three stimulus annotation styles: 1) "horiz_bar", which
        is a horizontal bar that denotes the duration of the stimulus, 
        2) "background_box", which is an underlaid box that denotes the
        duration of the stimulus, or 3) "double_bars" which is a pair of
        vertical bars that denote the start and end of the stimulus. 
    start (fl): Beginning of stimulus presentation in data coordinates.
    end (fl): End of stimulus presentation in data coordinates.
    top (fl): Applies only if `style`="horiz_bar". The top of the horizontal bar
        in data coordinates.
    bottom (fl): Applies only if `style`="horiz_bar". The bottom of the horizontal
        bar in data coordinates. 
    alpha (fl): The transparency of the stimulus annotation. Must be between 
        0 and 1. 
    level (str): The display level of the stimulus annotation, relative to the plot. 
        Default is "underlay". If `style`="background_box", can only be "underlay".
    
    Returns:
    --------
    `p` with stimulus annotation. 

    """

    # TODO: Add None option to pass default colours, which an be overriden by arg
    # TODO: "background_box" requires top and bottom as args, even though it doesn't use them; incorporate None

    assert (style == "horiz_bar" or "background_box" or "double_bars"), \
        f"{style} is not a valid entry for `style`. Please input either \
        'horiz_bar', 'background_box', or 'double_bars'."

    if style == "horiz_bar":
        bar = BoxAnnotation(left=start, 
                            right=end, 
                            top=top,
                            bottom=bottom,
                            fill_alpha=alpha, 
                            fill_color="#787878", 
                            level=level)
        p.add_layout(bar)

    elif style == "background_box":
        box = BoxAnnotation(left=start, 
                            right=end, 
                            fill_alpha=alpha, 
                            fill_color="#B6A290", 
                            level="underlay")
        p.add_layout(box)

    elif style == "double_bars":
        line_colour = "#775a42"
        start_line = Span(location=start, 
                          dimension="height", 
                          line_color=line_colour,
                          line_dash="dotted",
                          line_width=2) 
        end_line = Span(location=end, 
                        dimension="height",
                        line_color=line_colour,
                        line_dash="dotted",
                        line_width=2)
        p.add_layout(start_line)
        p.add_layout(end_line) 

    else:
        print()

    return p


def plot_aggregate_trace(df, group_by, val_col, time_col, 
                         val_label, time_label, 
                         palette, 
                         aggregation_method="mean", 
                         y_range=None,
                         legend_labels=None, theme=None,
                         mean_alpha=0.7, id_alpha=0.008, 
                         line_width=5, 
                         round_to=0, f_steps=1
                         ):

    """
    
    # TODO: ADD DOCS
    * N.B. Will ALWAYS make a legend. 

    """

    assert ("ID" in df), "The column 'ID' is not in in the input dataframe."
    assert ("trial" in df), "The column 'trial' is not in in the input dataframe."

    # TODO: Add None option for args for val_label and time_label
    # TODO: Add show option and save option

    p = figure(width=1000,
               height=400,
               y_range=y_range,
               x_axis_label=time_label,
               y_axis_label=val_label)
    
    grouped = df.groupby(group_by)

    for i, ((name,group), hue) in enumerate(zip(grouped, palette)):

        # Make dataframes:
        agg_df = aggregate_trace(group, 
                                 ["trial", time_col], 
                                 method=aggregation_method, 
                                 round_to=round_to, f_steps=f_steps
                                 ) 
        grouped_by_id = group.groupby(["ID"]) 

        assert len(palette)==len(grouped), \
            f"The lengths of `palette`, and `df.groupby({grouped})` must be equal."
        
        if legend_labels==None:
            legend_label = f"{name} | n={len(grouped_by_id)}"
        else:
            assert len(grouped)==len(legend_labels), \
                f"The lengths of `legend_labels`, `palette`, `df.groupby({grouped})` \
                must be equal."
            legend_label =  legend_labels[i]

        # Mean trace:
        p.line(x=agg_df[time_col], y=agg_df[val_col], 
               legend_label=legend_label, 
               color=hue,
               line_width=line_width,
               alpha=mean_alpha)

        # ID traces:
        for _, id_group in grouped_by_id:
            
            p.line(x=id_group[time_col], y=id_group[val_col], 
                   color=hue,
                   line_width=1, 
                   alpha=id_alpha, 
                   legend_label=legend_label,  
                   level="underlay")

    if theme==None:
        pass
    else:
        load_plot_theme(p, theme=theme, has_legend=True)
    
    return p 