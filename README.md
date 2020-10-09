# chemetho

An open-source Python toolkit for analyzing and visualizing **chemical ethology** data.

Written for analyzing the following data streams:

- Behavior from [FicTrac](https://github.com/rjdmoore/fictrac) and [Anipose](https://github.com/lambdaloop/anipose)
- Chemistry from photoionization detectors (PIDs) , and Shimadzu GC-MS and GC-FID instruments 

Uses `Bokeh` to generate interactive plots. 

This package was originally written to analyze data streams from [NOEXIIT](https://github.com/hanhanhan-kim/noexiit). In the NOEXIIT set-up, the data for FicTrac and Anipose are synchronously acquired via an external hardware trigger, and the PID data is independently streamed in parallel via a [DAQ](https://labjack.com/products/u3). 

## Installation

There are no current plans to upload this package to PyPI. Instead, install this package from source:

```bash
git clone https://github.com/hanhanhan-kim/chemetho.git
cd chemetho
pip install .
```

## How to use

This code is intended to be used with data that's sorted in a particular file tree structure:

`<date of acquisition> / <animal number> / <trial number> /`

Within each `/<trial>`, there should be three subdirectories, one for each data type: 

1. `/fictrac` stores FicTrac data

2. `/pose` stores videos for DeepLabCut and Anipose

3. `/stimulus` stores data about the stimulus presentation
   - PID data is stored here, even if the PID data is not derived from a stimulus 

Running the `make_tree.py` script in terminal will generate the above three subdirectories. It will also move files into the corresponding subdirectory, according to identifiers in the files' names. 

If you do not have all three of the above data types for your experiment, make only those subdirectories that are relevant to your experiments. 

#TODO: Provide an example file tree structure. 

#TODO: Decide whether to put GC-MS and GC-FID data as a 4th (general GC subdir?) or 4th and 5th subdirectory, even though they're not timeseries data. The alternative is to put them in their own thing. 

### FicTrac

Run the  `batch_fictrac.py` script in terminal to batch generate FicTrac data outputs from videos. 

To generate a selection of EDA plots from the FicTrac data output, enter the specified values in the `.yaml` file. See #TODO ADD DOCS the documentation on FicTrac analyses here, 

### Pose

#TODO: add docs

### Stimulus

#TODO: add docs



#TODO: Talk about Jupyter notebooks--maybe also include sample data for those notebooks to run 







