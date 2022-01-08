# Guided Internet-delivered Cognitive Behavioral Therapy Adherence Forecasting

#Dataset
The folder "Dataset" contains the dataset use in this work and made available as a benchmark for future research on
adherence forecasting for Internet Delivered Psychological Treatments.

The participants indexes used for hyperparameters selection and architecture building are:
[45, 18, 299, 113, 42, 228, 142, 106, 65, 61, 264, 284, 83, 231, 133, 56, 118, 272, 92, 278, 290, 185, 285, 328, 338,
206, 111, 136, 157, 304, 204, 82, 123, 72, 26, 305, 164, 5, 273, 70, 137, 200, 242, 46, 20, 35, 171, 27, 47, 213, 119, 
139, 16, 263, 9, 4, 301, 96, 76, 160, 236, 32, 218, 337, 319, 168, 309, 224, 80, 73, 85, 241, 266, 203, 140, 29, 220,
336, 19, 239, 268, 88, 64, 233, 227, 30, 101, 57, 167, 235, 252, 186, 54, 14, 128, 23, 182, 208, 317, 314]

The participants associated with these indexes should **not** be used when evaluating a classifier.

# Required Librairies

For training and evaluation: Pytorch {used version 1.9} (https://pytorch.org/), Pandas {used version 1.3.2} 
(https://pandas.pydata.org/), Scikit-Learn {used version 0.24.2} (https://scikit-learn.org/stable/),
Numpy {used version 1.20.3} (https://numpy.org/), SciPy {used version 1.6.2} (https://scipy.org/)

To generate the plots: Seaborn {used version 0.11.2} (https://seaborn.pydata.org/)

#Project Structure

The folder "Classification" contains both the hyperparameter search (classify_self_attention_variable_length.py) and the
training and testing of the best found model (classify_self_attention_variable_length.py)

The folder "AnalysisAndGraph" contains the python script to generate the graphs presented in this work's paper
(results_graph_generation.py), based on the results obtained (which are available in the folder "Results").

Finally, the folder "AblationSingleLength" contains the hyperparameter search (hyperparameters_search_single_length.py)
and the training and testing (classify_self_attention_single_length.py) of the network when considering only a single
sequence length (e.g. learning to predict adherence based on using only 11 days). 
