# X_Ray Covid State Classifier
## Introduction to a problem
The task is to build a classifier on x-ray chest to infer either or not a person has a COVID case.
## Data Description
There are around 10 000 images distributed more or less equaly between two classes: COVID and Non-COVID.
## Methodology
For this project I tried to build my own CNN architercture but transfer learning aproach was way better from the begining so final model had MobileNetV2 layer wich gave astonishing result of 97% accuracy on a test set.

## Conclusion 
Though results are very promising the real-life deployment of model might not be feasiable, its not certain how model will behave facing new diseases like tuberculosis and etc. But even as it is the model could help doctors better learn the differences between the COVID and Non-COVID cases. 
## P.S
Link for downloading data: https://data.mendeley.com/datasets/8h65ywd2jr/3