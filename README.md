# Cryptocurrencies
Module 18: Unsupervised Machine Learning and Cryptocurrencies

## Overview
We are helping Martha cluster and classify different cryptocurrencies so that she can help produce a report for her investment firm as they start progressing in the crypto world. They want to offer a new cryptocurrency investment portfolio for clients but they needed some help on visualizing and classifying cryptocurrencies. 
- We have a very limited data set, and because we don't have a clear output or target feature, we will use unsupervised machine learning clustering algorithms to help show the crypto data. 

## Analyzing the Data 
For our unsupervised learning, we will be conducting Principal Componenet Analysis (PCA). PCA is a super helpful statistical technique that will help speed up our machine learning algorithm. Below is the first 10 rows of the data set provided. PCA is used typically when there are a large amount of features, even though this doesn't look like a lot of features, since we are dealing with machine learning we will need to preprocess the data and turn the object/string values into numbers by encoding, thus drastically increasing the amount of features we will be using. 
![crypto_data](https://user-images.githubusercontent.com/79118630/124639940-bdf05800-de5a-11eb-9ce6-c4976fc7dd7f.png)

### Step 1: Preprocess the Data for PCA
- Part of the data we can see that there is a column called 'IsTrading' with True and False values. Since we are only dealing with active crypto, we only need to keep the True values. So we filter the data to just keep the True values, and drop the column. Then the next step was to fix the index. The column to the far left looks to be unnamed and has the coins abbreviations. So we set this column to be the index and drop the name. We can also see that there are some null values, so we must drop these rows because we only want clear and full data instances. 
- The final step before encode is to make sure that we only observe the currencies that have been mined. This seems a bit over the top since one would think that the currency must have been mined if its in the dataset. But looking at our results, there are over 150 rows that have a value less than 0 in the 'TotalCoinsMined' column. Now we are finally ready to encode. 
![TotalCoinsMined](https://user-images.githubusercontent.com/79118630/124640952-2429aa80-de5c-11eb-8853-fa605a2dc969.png)
- Before we encode, we will drop and take the 'CoinName' column as its own data frame because we will need it for later but won't need it for now. Now we only need to encode two columns, 'Algorithm' and 'ProofType', after we encode we are now given a total of 98 columns. Then we scale the data to make it easier for the machine learning algorithm.
![encoded_crypto_cols](https://user-images.githubusercontent.com/79118630/124641830-20e2ee80-de5d-11eb-9657-791860504af7.png)

### Step 2: Reduce Data Dimensions Using PCA
- For our study, we will be using PCA with three principal componenets. A key part to this is to make sure we keep the same index so the data stays clean and accurate.

![pca_df](https://user-images.githubusercontent.com/79118630/124642413-d150f280-de5d-11eb-80a1-10df678014b9.png)

### Step 3: Clustering Cryptocurrencies Using K-Means
- K-means is an unsupervised learning algorithm used to identify and solve clustering issues. Since we are not directly loooking at an output with unsupervised, we cluster our data, or more so the data clusters itself to show trends. Sometimes it is easy to spot how many clusters in a dataset, but our data is too messy/cluttered, so we will use an elbow curve in order to find the best value of k (amount of clusters) that we want to use to analyze the data.
![crypto_pca_elbow](https://user-images.githubusercontent.com/79118630/124643206-cfd3fa00-de5e-11eb-8b55-9fd0ebee9e8b.png)
- The elbow curve is called the elbow curve not only because it bends like an elbow on an arm, but we select the k-value where the line mostly bends like an elbow. Here it is clear that 4 is the best option, especially since it is almost a sharp turn at 4. Sometimes elbow curves can look this nice but sometimes it takes trial and error to see which one is the best.
- Next was to initialize the k-means model, fit it, and then run it to predict the clusters. 
- Finally, we take our original dataframe (from the end of the first step), merge with the PCA dataframe, and then merge the CoinName dataframe along with adding the predicted cluster class column
![combined_df_class](https://user-images.githubusercontent.com/79118630/124644608-766cca80-de60-11eb-8968-24d28efc27fa.png)

### Step 4: Visualizing Cryptocurrencies Results
- The first visualization we will use will be a 3d model. This way we can get a better picture on how the algorithm really clustered the data points. 
![3d_clustering](https://user-images.githubusercontent.com/79118630/124644942-d06d9000-de60-11eb-8a7c-0523ae40f1ec.png)
- We can see the 4 different clusters by the shapes and the different colors. As shown by the screenshot, each data point has a hover text box, with the coin name at the top in bold, the 3 principal components used to plot it and which mining algorithm was used for that coin. This 3d scale is helpful though because we can see those points in class 3 and class 2 are sort of outliers in a sense. The Class 3 points are much higher on the z-axis than classes 0 and 1, and the Class the 2 data point is far and to the left on the x-axis than those 2 classes. 
- We can also scale the 'TotalCoinSupply' and 'TotalCoinsMined' columns and plot those on a 2d scatter plot to visualize the data. Since we kept the same index throughout the whole project, the index will stay the same, thus keeping all the data points in the same classes.

![2d_crypto_scatter](https://user-images.githubusercontent.com/79118630/124645883-f6dffb00-de61-11eb-9679-5f4014f39415.png)
- These data points also have the hover feature, it is just not shown in this screenshot just to ensure it didn't look messy. Similar story is shown here, even though usually when you cluster data, there aren't a lot of overalapping points, but again there doesn't seem to be a lot of class 2 and 3 data points. Though this time, only class 2 seems to be the outlier since it is in the top right, and the class 3 point is just bunched with everybody else in the bottom right. 

## Conclusion
- We are now done helping Martha with her machine learning algorithm and she is ready to present the cryptocurrency data. We started off with over a 1,100 rows of data, but after the preprocessing, which was necessary, we are left with only 532 rows, or in other words, 532 tradable cryptocurrencies. 532 rows of data isn't a whole lot, so machine learning might not be the best use for analyzing this data. However, since the nature of our porjet was just to show grouping and clusters crypto data and since we didn't really have a 'target', unsupervised machine learning was perfect for this. 
- One thing Martha could do is since we kept the index the same, look at those funky outliers and see why they are there. Maybe even remove them and see what happens. She could also try adding another cluster, to make k equal 5. Having k=4 seemed like the best option from the elbow curve plot, but there was a little curve at 5. The good thing is that she has the code now, so she could easily go in and just change it to see what the new outcome would be and if that would make any notable changes. 




