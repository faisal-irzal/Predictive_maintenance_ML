# Bearing Anomaly Detection using Machine Learning

## 1. Introduction
Condition monitoring of an equipment is one of the crucial aspect one must conduct while operating a plant. The program can help identify the health state of the equipment so the plant operator can make necessary action to mitigate the equipment failure.

The most common way to perform condition monitoring is to look at each sensor measurement from the machine and to set a minimum and maximum value limit on the measurement. If the measurement value is within the bounds, then the equpment is healthy. If the measurement value is outside the bouds, then the equipment is unhealthy and a warning is sent.

Due to signals are measured from many sensors, the system is prone to either sending a large number of false alarms, i.e. sending unnecessary alarms or missing alarms, i.e. not sending required alarms. In order to mitigate this situation, it is necessary to have a reliable computational method which can analyze and give correct judgement from the combined data collected from various measurements of the available sensors.

In recent years many novel methods have been introduced to help plant operator correctly analyze the combined signals, thanks to the rapid machine learning and deep learning application to the industry. 

I recently came across a great [article](https://towardsdatascience.com/how-to-use-machine-learning-for-anomaly-detection-and-condition-monitoring-6742f82900d7) written by Vegard Flovik where he highlighted two different approaches of unsupervised machine learning methods to detect anomaly of a working machine to help operator monitor its condition. In the article, he introduced two different types of unsupervised learning methods to solve the anomaly detection problem and provided practical code example;
* Principal Component Analysis (PCA) combined with Mahalanobis distance
* Autoencoder

For the sake of understanding the introduced concept in the article, I have followed his code and tried to reproduce the results.


## 2. Data

The bearing dataset was provided  provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati and is available at [Kaggle](https://www.kaggle.com/vinayak123tyagi/bearing-dataset). There exists 3 datasets in total and each dataset consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Following the article, the second dataset was selected to perform this study.

For the second dataset, there exists 984 data files with a measurement interval of 10 minutes between each two adjacent data file, and each data file has 20,480 measurements. This means a measurement exercise of the equipment was conducted every 10 minutes and in each exercise 20,480 measurements were taken.

For each measurement, there are 4 different parameters recorded, representing 4 different types of bearings of the equipment. This project is essentially studying the interactions between the 4 bearings. For each data file, Vegard Flovik took an average of all 20,480 measurements and the final dataset has 984 rows and 4 columns. I believe the purpose of taking the average is for simplicity and the reduction of noise among individual measurements. However, one can actually train a more sophisticated model by using the large number of individual measurements without taking the average.

Plotting the raw data with respect to its index (time of observation) we get the following plot

