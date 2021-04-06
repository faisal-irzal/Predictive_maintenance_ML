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

## 3. Define train/test data

Before setting up the models, we need to define train/test data. To do this, we perform a simple split where we train on the first part of the dataset (which should represent normal operating conditions), and test on the remaining parts of the dataset leading up to the bearing failure. Plot below shows the train dataset we use to train the model.

![Screenshot 2021-04-06 at 15 32 28](https://user-images.githubusercontent.com/76395229/113718990-7b00d600-96ed-11eb-99a4-cbf4790a8c6c.png)

## 4. Approach-1 Multivariate Statistical Analysis (MSA)

Having split the dataset into train and test data, we will now build a model that can detect anomaly from the bearing datasets. This section will be split into two subsection; first is building the model using PCA, second is anomaly detection using Mahalanobis distance.

### 4.1 Principal Component Analysis (PCA) application to the MSA

Dealing with high dimensional data is often computationally challenging. Luckily there are several techniques to reduce the number of variables. One of the main techniques we are going to use here is the Principal Component Analysis (PCA). 

Here, PCA performs a linear mapping of the data to a lower-dimensional space in such a way that the variance of the data in the low-dimensional representation is maximized. In practice, the covariance matrix of the data is constructed and the eigenvectors of this matrix are computed. The eigenvectors that correspond to the largest eigenvalues (the principal components) can now be used to reconstruct a large fraction of the variance of the original data. The original feature space has now been reduced to the space spanned by a few eigenvectors.

PCA model has been built based on the training data and features have been reduced to 2 principal components. It is observed that the principal component 1 holds 51.0% of the information while the principal component 2 holds only 20.4% of the information. Also, the other point to note is that while projecting four-dimensional data to a two-dimensional data, 28.6% information was lost.


### 4.2 Anomaly Detection using Mahalanobis distance

The Mahalanobis distance is widely used in cluster analysis and classification techniques. In order to use the Mahalanobis distance to classify a test point as belonging to one of N classes, one first estimates the covariance matrix of each class, usually based on samples known to belong to each class. In our case, as we are only interested in classifying “normal” vs “anomaly”, we use training data that only contains normal operating conditions to calculate the covariance matrix. Then, given a test sample, we compute the Mahalanobis distance to the “normal” class, and classifies the test point as an “anomaly” if the distance is above a certain threshold. More about Mahalanobis distance, please follow this [link](https://www.machinelearningplus.com/statistics/mahalanobis-distance/).

The square of the Mahalanobis distance to the centroid of the distribution should follow chi-square (χ2) distribution if the assumption of normal distributed input variables is fulfilled. This is also the assumption behind the above calculation of the “threshold value” for flagging an anomaly. As this assumption is not necessarily fulfilled in our case, it is beneficial to visualize the distribution of the Mahalanobis distance to set a good threshold value for flagging anomalies.

![Screenshot 2021-04-06 at 15 42 12](https://user-images.githubusercontent.com/76395229/113720249-bc45b580-96ee-11eb-8a17-cc3964ba66c1.png)

From the distribution above we can set the anomaly threshold equals to 4 standard deviations from mean of training data's Mahalanobis distances. Mahalanobis distance of the training data, threshold value and anomaly flag variable for both train and test data can be saved into a dataframe.

![Screenshot 2021-04-06 at 15 43 39](https://user-images.githubusercontent.com/76395229/113720495-fca53380-96ee-11eb-8715-bd2627c5c658.png)

The chart above shows that anomaly can be detected between 2004-02-16 to 2004-02-17 without the risk of missing Bearing 1. That's practically 2 days before when the breakdown actually occurred, on 2004-02-19.

## 5. Approach 2: Autoencoder model for anomaly detection

An autoencoder is a type of artificial neural network used to learn efficient data encodings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input.

In this case study, the distribution of the reconstruction loss of the autoencoder for training data is plotted to identify where normally reconstruction loss lies and come up with a threshold as the upper control limit. One can also compute the 3 standard deviations from the mean of reconstruction loss to determine an appropriate upper control limit, similar to what have been done in the first modeling approach above.

### 5.1 Define and train the model

The ELU (Exponential Linear Unit) activation function is used, which returns x if x > 0 and exp((x)-1) if x < 0). Such choice allows faster convergence of model weigths compared to RELU because of non-null output for x < 0. No difference was observed based on variation of the initialization of weights. This may mean that model training is complete after 100 iterations, and thus results are stable.

![Screenshot 2021-04-06 at 16 03 02](https://user-images.githubusercontent.com/76395229/113723412-9ec61b00-96f1-11eb-93db-ba36e1b0af97.png)


### 5.2 Distribution of loss function in the training set

By plotting the distribution of the calculated loss in the training set, one can use this to identify a suitable threshold value for identifying an anomaly. In doing this, one can make sure that this threshold is set above the “noise level”, and that any flagged anomalies should be statistically significant above the noise background.

From our result, we can use a threshold of 0.3 for flagging an anomaly. We can then calculate the loss in the test set, to check when the output crosses the anomaly threshold.

![Screenshot 2021-04-06 at 16 05 24](https://user-images.githubusercontent.com/76395229/113723762-f6648680-96f1-11eb-8427-de790402045a.png)

From the above loss distribution, a threshold of 0.3 is set for flagging an anomaly. We can then calculate the loss in the test set, to check when the output crosses the anomaly threshold.

![Screenshot 2021-04-06 at 16 06 59](https://user-images.githubusercontent.com/76395229/113723959-290e7f00-96f2-11eb-8f0f-6fca6837d024.png)

Similar to the PCA with Mahalanobis distance method, the chart above shows that the anomaly can be detected on 2004-02-16. One can see that it detects anomaly slightly earlier than the PCA with Mahalanobis distance model. This may due to the choice of the hyperparameter of the model and may also due to Autoencoder doesn't have the assumption that the input data follows Gaussian distribution, which is a constraint of Mahalanobis distance. This requires further investigation.

## 6. Conclusions

Two modelling methods have been discussed to analyze data collected from various measurements of the sensors as means of machine monitoring condition. Both methods showed similar results where they are able to predict the anomaly prior to the actual failure. The main difference between two approaches lays in the definition of a suitable threshold value for anomalies. Based on the results discussed above, the approach using autoencoder of neural network is slightly superior to the one using a combination of PCA and Mahalanobis distance. 



