# visual-cortex-response-classification

This repository contains the code to reproduce the results in our paper titled "Functional Parcellation of Mouse Visual Cortex Using Statistical Techniques Reveals Response-Dependent Clustering of Cortical Processing Areas"

## Abstract
The visual cortex has a prominent role in the processing of visual information by the brain. Previous work has segmented the mouse visual cortex into different areas based on the organization of retinotopic maps. Here, we collect responses of the visual cortex to various types of stimuli and ask if we could discover unique clusters from this dataset using machine learning methods. The retinotopy based area borders are used as ground truth to compare the performance of our clustering algorithms. We show our results on two datasets, one collected by the authors using wide-field imaging and another a publicly available dataset collected using two-photon imaging. The proposed supervised approach is able to predict the area labels accurately using neuronal responses to various visual stimuli. Following up on these results using visual stimuli, we hypothesized that each area of the mouse brain has unique responses that can be used to classify the area independently of stimuli. Experiments using resting state responses, without any overt stimulus, confirm this hypothesis.  Such activity-based segmentation of the mouse visual cortex suggests that large-scale imaging combined with a machine learning algorithm may enable new insights into the functional organization of the visual cortex in mice and other species.

## Orgnaization
- As mentioned in the abstract, we show results on two different dataset, one collected using two-photo imaging and another collected using wide-field imaging.
- The demo of our code for repeating our results using the two-photon imaging can be found [here](two-photon/ReadMe.md)
- The demo of our code for repeating our results using the wide-field imaging can be found [here](wide-field-imaging/ReadMe.md)

## Requirements

- python (version 3 or above)
- MATLAB (version R2018a or above)

### Required Python Packages
- allensdk
- scipy
- numpy
- pandas
- seaborn
- tqdm
- jupyterlab
- ipywidgets
- MATLAB Engine API for Python
#### Note: Instruction for installing MATLAB Engine API for Python can be found [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
