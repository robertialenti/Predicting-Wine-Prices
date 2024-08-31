# Predicting Wine Prices
This project aims to predict prices for a selection of wines reviewed by sommeliers. I compare the performance of naive and linear regression models to a number of commonly used machine learning techniques including lasso regression, K-nearest neighbors, random forest, and two gradient-boosted decision trees (XGBoost and LightGBM).

## Data
The data was scraped by [Zach Thoutt](https://github.com/zackthoutt/wine-deep-learning) in 2017 from [Wine Enthusiast](https://www.wineenthusiast.com/?s=&search_type=shop). The raw dataset includes nearly 150,000 reviews. For each wine, we have information about its production, its country and region of origin, a review written written by a professional sommelier, the score assigned to it by Wine Enthusiast, as well as its price in US dollars.

## Code
The code is separated into 5 sections. To run the code without modification, I suggest creating a project directory with `code`, `data`, `figures`, and `output` folders. 

### 1. Preliminaries
I begin by importing widely used libraries for data analysis, machine learning, and natural language processing. 

### 2. Importing and Cleaning Data
After reading in the data, I create a new variable, `vintage` by parsing the vintage year from the `title` variable. I then subset the data, focusing only on vintages between 1990 and 2016 - which comprise nearly the entirety of the dataset. I retain only wines with strictly positive prices. I also interpolate missing values of `region_1` by using the designation-specific modal value. Finally, I remove unneeded variables, or variables that I deem to have little predictive power.

The processed dataset includes just over 107,000 observations. Variables include:

- points: The number of points awarded by Wine Enthusiast, on a scale from 1 to 100.
- variety: The type or types of grapes used in a wine's production.
- description: A sommelier's review. 
- country: The wine's country of origin.
- province: The wine's province or state of origin.
- region_1: The wine's growing area. This is often, but not always, the name of the wine's area of protected origin.
- winery: The wine's producer. 
- designation: the vineyard within the winery where the grapes that made the wine are from
- vintage: The vintage of the wine, ranging from 1990 to 2016.
- price: The cost of the wine, in US dollars.

### 3. Natural Language Processing
I evaluate the sentiment of each description by calculating polarity scores with Vader's sentiment analysis library. 

In addition, I parse the sommelier reviews for the use of technical terms with high predicive power. More specifically, I tabulate how many unique (non-Stop words) are used by sommeliers in their reviews and retain the 1000 most common. For each, I create an indicator variable, equal to 1 if a review includes the term, and 0 otherwise. I then correlate the indicator with price. I repeat the process for all 1000 words, retaining only the 10 words most highly correlated (either positively or negatively) with wine prices. This includes terms such as: tannin, bold, concentrated, smooth, and dark, among others, which are generally used to describe wines of higher value.

### 4. Exploratory Analysis
In this section, I visualize some of the data's features to get a better sense of how key variables are distributed and related.

First, I plot the country of origin. Most of the wines in the dataset were produced in the United States.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/country_distribution.png" width="500" height="300">

Next, I plot the distribution of grape variety. Consistent with global production, pinot noir, chardonnay, and cabernet sauvignon are the most common grape varieties. However, a large number of wines are formed from less common blends.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/variety_distribution.png" width="500" height="300">

Finally, I plot the distribution of vintage. Most of the wines in the dataset were produceed from grapes harvested around 2010. Given that data was gathered in 2017, most of the wines in the dataset are not young wines - which are consumed within 1-2 years of bottling - but rather wines that have already aged for several years.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/vintage_distribution.png" width="500" height="300">

To get a better sense of how quantitative variables are related with price, I construct a correlation matrix.

I find that wine price to be left-skewed, with most wines having prices clustered around the mean and relatively fewer wines having much higher prices. As a result, I choose to apply a logarithmic transform, which helps make the distribution of price more normal. I use the logged price as the target variable in prediction.

### 5. Prediction
In this section, I select relevant features, create training and testing sets, parametrize the machine learning models with k-fold cross-validation, make out-of-sample predictions, and compare performance to a naive model.

The features selected include all of the numerical variables, namely ___, as well as categorical variables, including ___.

I create the training dataset by selecting a random sample of the processed dataset with 80% of the processed dataset's observation. The testing dataset comprises the complimentary 20% of wines.

In addition to a naive model, which simply assumes that a wine's price is equal to the average price, I train and use 6 additional models: linear regression, lasso regression, K-nearest neighbors, random forest, and gradient-boosted decision trees (XGBoost and LightGBM). The predictive performance, expressed in terms of mean absolute error (MAE) and mean absolute percentage error (MAPE) is shown below:


The naive model, which makes only unconditional predictions, is found to perform the worse. In comparison, the gradient-boosted decision trees are found to perform best. In particular, the XGBoost and LightGBM models are found to record MAPEs of just 7.81% and 8.11%, respectivey. That is, these models make predictions that are, on average, 8% away from a wine's actual log(price), which is more than twice as accurate as the naive model. XGBoost remains the best performing model if price, rather than log(price), is used as an outcome. However, the model's MAPE rises to nearly 30% when predicting the untransformed price variable.

| Model | MAE | MAPE (%) | MAPE, Relative to Naive |
| ----- | --- | -------- | ----------------------- |
| Naive | 0.53 | 16.30 | 1 |
| Linear Regression | 0.38 | 11.50 | 0.71 |
| Lasso Regression | 0.38 | 11.69 | 0.72 |
| K-Nearest Neighbors | 0.35 | 10.69 | 0.66 |
| Random Forest | 0.28 | 8.45 | 0.52 |
| XGBoost | 0.26 | 7.81 | 0.48 |
| LightGBM | 0.27 | 8.11 | 0.50 |

In addition, I plot actual and predicted log price values for the 6 machine learning models.

<img src="https://github.com/robertialenti/Wine/raw/main/output/predicted_actual_combined.png" width="500" height="300">

Finally, I examine feature importance for models that provide this information. As expected, Wine Enthusiasts' points are generally good predictors of price. A wine's grape country and region of interest, as well as its grape blend are also consistently strong predictors. The sommelier's reviews are comparatively less strong predictors.

Unfortunately, the original dataset does not contain the sommeliers' price assessments. As such, we cannot compare the performance of the models to the experts' best guess.
