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

![Screenshot 2021-04-06 at 15 26 35](https://user-images.githubusercontent.com/76395229/113718341-d383a380-96ec-11eb-83dd-142a62a85100.png)

From the plot above we notice that Bearing 1 started to deviate from original trend after 2004-02-16 while others still performed quite normal until after 2004-02-18. In traditional Statistical Process Control (SPC), if we are unlucky, we may not have selected Bearing 1 to be monitored and may not have detected any issue before 2004-02-19, when a breakdown event occurred.

## Define train/test data

Before setting up the models, we need to define train/test data. To do this, we perform a simple split where we train on the first part of the dataset (which should represent normal operating conditions), and test on the remaining parts of the dataset leading up to the bearing failure. Plot below shows the train dataset we use to train the model.

![Screenshot 2021-04-06 at 15 32 28](https://user-images.githubusercontent.com/76395229/113718990-7b00d600-96ed-11eb-99a4-cbf4790a8c6c.png)

