# Predicting Wine Prices
This project aims to predict prices for a selection of wines reviewed by sommeliers. I compare the performance of rule-of-thumb and multiple linear regression models to a number of commonly used machine learning techniques including lasso regression, K-nearest neighbors, random forest, and two gradient-boosted decision trees (XGBoost and LightGBM).

## Data
The data was scraped by [Zach Thoutt](https://github.com/zackthoutt/wine-deep-learning) in 2017 from [Wine Enthusiast](https://www.wineenthusiast.com/?s=&search_type=shop). The raw dataset includes nearly 150,000 reviews. For each wine, we have information about its production, its country and region of origin, the score assigned to it by Wine Enthusiast, as well as its price in US dollars.

## Code
The code is separated into 5 sections.

### 1. Preliminaries
I begin by importing widely used libraries for data analysis, machine learning, and natural language processing. 

### 2. Importing and Cleaning Data
I read in the data, __. I create a new variable, `vintage` by parsing the vintage year from the `title` variable. I then subset the data, focusing only on vintages between 1990 and 2016 - which comprise nearly the entirety of the dataset. I retain only wines with strictly positive prices. I also interpolate missing values of `region_` by ___. Finally, I remove unneeded variables, or variables that I deem to have little predictive power.

The processed dataset includes just over 107,000 observations. Variables include:

- Points: the number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)
- Variety: the type of grapes used to make the wine (ie Pinot Noir)
- Description: a few sentences from a sommelier describing the wine's taste, smell, look, feel, etc.
- Country: the country that the wine is from
- Province: the province or state that the wine is from
- Region 1: the wine growing area in a province or state (ie Napa)
- Region 2: sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank
- Winery: the winery that made the wine
- Designation: the vineyard within the winery where the grapes that made the wine are from
- Vintage: The vintage of the wine, ranging from 1990 to 2016.
- Price: The cost of the wine, in US dollars.

### 3. Natural Language Processing
I evaluate the sentiment of each description using a transformer, which is a deep learning model better suited to interpret human language.

In addition, I parse the sommelier reviews for mentions of terms with the most predictive power.

### 4. Exploratory Analysis
Before proceeding to making predictions, I visualize some of the data's features to get a better sense of how key variables are distributed and related.

First, I plot the country of origin. Most of the wines in the dataset were produced in the United States.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/country_distribution.png" width="425" height="250">

Next, I plot the distribution of grape variety. Consistent with global production, pinot noir, chardonnay, and cabernet sauvignon are the most common grape varieties. However, a large number of wines are formed from less common blends.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/variety_distribution.png" width="425" height="250">

Finally, I plot the distribution of vintage. Most of the wines in the dataset were produceed from grapes harvested around 2010. Given that data was gathered in 2017, most of the wines in the dataset are not young wines - which are consumed within 1-2 years of bottling - but rather wines that have already aged for several years.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/vintage_distribution.png" width="425" height="250">

Given that 

I find that wine price to be left-skewed, with most wines priced near the mean and relatively few wines with much higher prices. As a result, I choose to apply a logarithmic transform, which helps the distribution of price more normal and reduce the ___. I use the logged price variable in prediction.

### 5. Prediction
In this section, I select relevant features, create training and testing sets, parametrize the machine learning models with k-fold cross-validation, make out-of-sample predictions, and compare performance to a naive model.

The features selected include all of the numerical variables, namely ___, as well as categorical variables including ____.

I create the training dataset by selecting a random sample of the processed dataset with 80% of the processed dataset's observation. The testing dataset comprises the complimentary 20% of wines.

In addition to a naive model, which simply assumes that a wine's price is equal to the average price, I train and use 6 additional models: linear regression, lasso regression, K-nearest neighbors, random forest, and gradient-boosted decision trees (XGBoost and LightGBM). The predictive performance, expressed in terms of mean absolute error (MAE) and mean absolute percentage error (MAPE) is shown below:


The naive model, which makes only unconditional predictions, is found to perform the worse. In comparison, the gradient-boosted decision trees are found to perform best, recording MAPE of around 30%. That is, these models make predictions that are, on average, 30% away from a wine's actual price.

| Model | Price - MAE | Price - MAPE | Log(Price) - MAE | Log(Price) - MAPE |
| ---- | ---- | ------------ | -------- | --------- |
| Naive | 2018-04-23 17:47 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4744 | -73.604 |
| Linear Regression | 2018-04-23 18:00 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4744 | -73.604 |
| Lasso Regression | 2018-04-23 18:03 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4744 | -73.604 |
| K-Nearest Neighbors | 2019-06-10 17:45 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4739 | -73.6047 |
| Random Forest | 2019-06-10 17:49 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4739 | -73.6047 | 
| XGBoost | 2019-06-10 17:50 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4739 | -73.6047 |
| LightGBM | 2019-06-10 17:50 | Métro Vendôme (de Marlowe / de Maisonneuve) | 45.4739 | -73.6047 |

In addition, I plot actual and predicted log price values for the 6 machine learning models.

Unfortunately, the original dataset does not contain the sommeliers' price assessments. As such, we cannot compare the performance of the models to the experts' best guess.
