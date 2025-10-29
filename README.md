Project Description

This project aims to develop a machine learning system for the early and non-invasive detection of Parkinson's Disease (PD). PD is a neurodegenerative disorder that often manifests in vocal impairments (known as hypokinetic dysarthria). The system will analyze a set of 22 biomedical voice measurements from individuals. These features capture various aspects of vocal acoustics, such as fundamental frequency, variation in amplitude, and measures of nonlinear dynamics. By training on this data, the model will learn to classify subjects into two categories: healthy or Parkinson's positive. This tool has the potential to assist in early diagnosis and monitoring of the disease.

Dataset Link
https://www.kaggle.com/datasets/jainaru/parkinson-disease-detection

Dataset Details
•	Original Number of Features in the Dataset: 23 Columns (22 feature variables + 1 target variable)

•	Target Variable: status
o	1 - Indicates the presence of Parkinson's Disease.
o	0 - Indicates a healthy subject.

•	Dataset Variable Description:
o	MDVP:Fo(Hz): Average vocal fundamental frequency.
o	MDVP:Fhi(Hz): Maximum vocal fundamental frequency.
o	MDVP:Flo(Hz): Minimum vocal fundamental frequency.
o	MDVP:Jitter(%): A measure of period-to-period variation in fundamental frequency (percentage).
o	MDVP:Jitter(Abs): Absolute measure of jitter.
o	MDVP:RAP: Relative amplitude perturbation.
o	MDVP:PPQ: Five-point period perturbation quotient.
o	Jitter:DDP: Average absolute difference of differences between cycles, related to RAP.
o	MDVP:Shimmer: A measure of period-to-period variation in amplitude.
o	MDVP:Shimmer(dB): Shimmer measured in decibels.
o	Shimmer:APQ3: Three-point amplitude perturbation quotient.
o	Shimmer:APQ5: Five-point amplitude perturbation quotient.
o	MDVP:APQ: A different method of measuring amplitude perturbation.
o	Shimmer:DDA: Average absolute difference between consecutive differences between consecutive amplitudes, related to APQ3.
o	NHR: Noise-to-harmonics ratio.
o	HNR: Harmonics-to-noise ratio.
o	RPDE: Recurrence Period Density Entropy (a nonlinear dynamical complexity measure).
o	DFA: Detrended Fluctuation Analysis (a scale-invariant fractal measure of self-similarity).
o	spread1: A nonlinear measure of fundamental frequency variation.
o	spread2: A nonlinear measure of fundamental frequency variation.
o	D2: Correlation dimension (another nonlinear measure).
o	PPE: Pitch Period Entropy.

•	Type of Problem: Supervised - Classification (Binary Classification)

•	Algorithms Selected:
o	Algorithm 01: Support Vector Machine (SVM)
o	Algorithm 02: Random Forest Classifier

Colab Link
https://colab.research.google.com/drive/14zwn-Q-xueXGKLF_tBaVu3RkWJQPdUa0?usp=sharing
GitHub Repository
https://github.com/pamudithasandaru/Parkinson-s-Disease-Detector.git

