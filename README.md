# Analysis pipeline for the Human MEEG experiment

The analyses are done with Python 3.7, based on the package MNE python. For a reference article on the package, see:

	M. Jas, E. Larson, D. A. Engemann, J. Leppäkangas, S. Taulu, M. Hämäläinen, A. Gramfort (2018).
    A reproducible MEG/EEG group study with the MNE software: recommendations, quality assessments,
    and good practices. Frontiers in neuroscience, 12.

Additionally, you will need to install the autoreject package that is a machine-learning based algorithm to identify outlier
epochs and automatically interpolate or reject the bad epochs. All the information about the package is here:
[link to package](https://autoreject.github.io/)

and the corresponding article 

    Mainak Jas, Denis Engemann, Yousra Bekhti, Federico Raimondo, and Alexandre Gramfort. 2017. “Autoreject: Automated artifact rejection for MEG and EEG data”. NeuroImage, 159, 417-429.

# MEEG preprocessing steps

[config.py](config.py) | The config file contains the paths to the data, the results folder, the scripts. It also contains all the parameters 
related to the preprocessing: baselining, filtering, temporal windows for epoching. To adapt the config file, all specific settings to be used in your analysis are defined in [config.py](config.py).
See the comments for explanations and recommendations. 

The following preprocessing steps are specific to MEEG data analysis:

[00-review_raw_data_for_bad_channels.py](00-review_raw_data_for_bad_channels.py) | 
[01-import_and_filter.py](01-import_and_filter.py) | 
[02-apply_maxwell_filter.py](02-apply_maxwell_filter.py) | 
[03-run_ica.py](03-run_ica.py) | 
[04-identify_EOG_ECG_components_ica.py](04-identify_EOG_ECG_components_ica.py) | 
[05-apply_ica.py](05-apply_ica.py) | 

# Temporal segmentation of the data (epoching) and plotting the evoked responses

MNE python is particularly suited for MEG and EEG data analysis but it is rather easy to build MNE-compatible data objects and then use all the
functions provided by the package. For a tutorial, see:

[Link to tutorial](https://mne.tools/stable/auto_examples/io/plot_objects_from_arrays.html#sphx-glr-auto-examples-io-plot-objects-from-arrays-py)

[06-make_epochs.py](06-make_epochs.py) | Will build epochs objects without removing the bad ones or using autoreject to identify, interpolate or remove the bad epochs.
[07-sanity_check_plots.py](07-sanity_check_plots.py) | Plots for every participant separately the evoked responses and the global field power (GFP).

# Effect matched spatial filter

	Aaron Schurger, Sebastien Marti, and Stanislas Dehaene, “Reducing multi-sensor data to a single time course that 
	reveals experimental effects”, BMC Neuroscience 2013, 14:122.

[Link to tutorial](https://mne.tools/dev/auto_examples/decoding/plot_ems_filtering.html)

# Linear regressions and residual analyses

To model the surprise from transition probabilities, we used an ideal observer from the package MarkovModel_Python

[link to package](https://github.com/florentmeyniel/TransitionProbModel)

