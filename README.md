# ATIV

Welcome to Adaptive Thermal Image Velocimetry. This package is designed to give the user a quick access to run A-TIV and TIV algorithms. 

Current Version: 0.1


## 1. Introduction

Thermal Image Velocimetry is an optical flow algorithm, based on particle image velocimetry techniques designed to estimate spatial wind velocities from thermal video. The algorithm was initially described by Inagaki (2013) who used particle image velocimetry techniques of Kaga (1992) on thermal video of thermally responsive artificial surfaces to estimate spatial wind velocities. The technique was further investigated by Schumacher (2019) on artificial created patterns to test the user input settings. With this package we are presenting now the evolution of the TIV algorithm A-TIV which allows to retrieve spatial velocities over a wider range of surface types with less user-input needed. 

## 2. Setup

At the current stage there is unfortunately no quick setup available. We are currently working on this to provide an updated version with a setup file.

For now please follow the instructions:

0. clone this repository to your computer

1. Install the following packages:

- numpy
- scipy 
- progressbar2
- joblib
- Py-EMD
- collections
- statistics


## 3. Use

Load the stabilized IR video to a numpy array (dimensions: time, x, y). Then use the example file to calculate A-TIV.
Use Schumacher 2019 to set the algorithm parameters.  Generally it can be stated that the parameters are dependent on the wind velocity, on the quality of the input IR video and on the desired output resolution.

Here is a list of the input parameters:

ws = 32
The search window size in pixels to estimate the vector (the smaller it gets the longer it takes to calculate one image)

ol = 28
The overlap of the search windows in pixels within the search area. Maximum value is ws-1 (31 in this case). This defines the density of the vectors calculated, when this parameter is defined higher it takes longer to calculate one image.

sa = 64
The search area size in pixels defining the maximum search area around the fixed window in image 1. Indirectly defining the maximum wind speed which can be resolved.

olsa=62
The search area overlaps

method = "greyscale"
One of the following: greyscale, ssim, rmse
The greyscale technique is currently in a sensitivity study the most accurate (Schumacher 2019).


time_lst=[60, 40, 20, 10]
The time list in frames (in fps) which the perturbations are calculated on. Optional the spatiotemporal, in example 1 the temporal option is used. 
Recommendation: Set a list of 4-5 times including 5 - 30 seconds of perturbations. If the length of the list is 1 then a TIV will be calculated.


time_interval = 3
The time increment between the images, this variable is informed by the HHT in the ATIV.

set_len = 4
A parameter to limit the amount of images calculated. In example 1 set to 4 to calculate the first 4 A-TIV outputs.


## 4. Algorithm

The Algorithm partially uses Liberzon(2021) in the calculation of the window locations and the sub-pixel peaks, however the window correlation of this package differs substantially. Furthermore this package offers 3 different correlation methods compared to PIV implementations which use FFT based cross correlation.


## 5. References

A. Inagaki, M. Kanda, S. Onomura, and H. Kumemura.  Thermal Image Velocimetry.Boundary-527Layer Meteorology, 149(1):1–18, 2013.  doi:  10.1007/s10546-013-9832-z.

Kaga, A. and Inoue, Y. and Yamaguchi (1992): Application of a Fast Algorithm for Pattern tracking on Airflow Measurements.

Alex Liberzon; Theo Käufer; Andreas Bauer; Peter Vennemann; Erich Zimmer. OpenPIV/openpiv-python: OpenPIV-Python v0.23.4. doi: https://doi.org/10.5281/zenodo.4409178

B. Schumacher, M. Katurji, J. Zhang, I. Stiperski, and C. Dunker.  Evolution of micrometeorolog-562ical observations instantaneous spatial and temporal surface wind velocity from thermal image563processing.Geocomputation Conference 2019, 2019.  doi:  10.17608/k6.auckland.9869942.v1

## 6. How to cite

Benjamin Schumacher (2021). Adaptive Thermal Image Velocimetry: ATIV v0.1. doi: https://doi.org/10.5281/ZENODO.4741550
