# Predicting Wine Prices
This project aims to predict prices for a selection of wines reviewed by sommeliers and listed on Wine Enthusiast. I compare the performance of a naive model to a linear regression model, as well as to a number of commonly used machine learning techniques including lasso regression, K-nearest neighbors, random forest, and two gradient-boosted decision trees (XGBoost and LightGBM).

## Data
The raw data was scraped by [Zach Thoutt](https://github.com/zackthoutt/wine-deep-learning) in 2017 from [Wine Enthusiast](https://www.wineenthusiast.com/?s=&search_type=shop). The raw dataset includes nearly 150,000 reviews. For each wine, the dataset provides information about its country and region of origin, its designation, grape blend, producer, as well as a review written written by a professional sommelier, the score assigned to it by Wine Enthusiast, and its price in US dollars.

## Code
The code is separated into 5 sections. To run the code without modification, create a project directory with `code`, `data`, `figures`, and `output` folders. 

### 1. Preliminaries
I begin by importing widely used libraries for data analysis, machine learning, and natural language processing. 

### 2. Importing and Cleaning Data
After reading in the data, I create a new variable, `vintage` by parsing the vintage year from the `title` variable. I then subset the data, focusing only on vintages between 1990 and 2016 - which comprise nearly the entirety of the dataset. I retain only wines with strictly positive prices. I also interpolate missing values of `region_1` by using the designation-specific modal value. Finally, I remove unneeded variables, or variables that I deem to have little predictive power. The processed dataset includes just over 107,000 observations.

### 3. Natural Language Processing
I evaluate the sentiment of each description by calculating polarity scores with [Vader's](https://github.com/cjhutto/vaderSentiment) sentiment analysis library. 

In addition, I parse the sommelier reviews to identify technical terms with high predicive power. More specifically, I tabulate how many unique (non-stop words) are used by sommeliers in their reviews and retain the 1000 most common. For each, I create an indicator variable, equal to 1 if a review includes the term, and 0 otherwise. I then correlate each indicator with price, retaining only the 10 words that are most highly correlated (either positively or negatively) with wine prices. 

### 4. Exploratory Analysis
In this section, I visualize some of the data's features to get a better sense of how key variables are distributed and related with price.

First, I plot the country of origin. Most of the wines in the dataset were produced in the United States.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/country_distribution.png" width="600" height="400">

Next, I plot the distribution of grape variety. Consistent with global production, pinot noir, chardonnay, and cabernet sauvignon are the most common grape varieties. However, a large number of wines are formed from less common blends.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/variety_distribution.png" width="600" height="400">

Finally, I plot the distribution of vintage. Most of the wines in the dataset were produceed from grapes harvested around 2010. Given that data was gathered in 2017, most of the wines in the dataset are not young wines - which are consumed within 1-2 years of bottling - but rather wines that have already aged for several years.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/vintage_distribution.png" width="600" height="400">

To get a better sense of how quantitative variables are related with price, I construct a correlation matrix. Unsurprisingly, points and sentiment are positively related with price. Vintage is negatively correlated with price, given that older wines typically fetch higher prices on account of having more complex flavors. Some terms are positively related with price, including "tannins", "bold", "concentrated", and "dark", among others. The term "fruity" is negatively correlated with price, as ___ are typically perceived as being less sophisticated.

<img src="https://github.com/robertialenti/Wine/raw/main/figures/correlation_matrix.png" width="600" height="450">

I find price to be left-skewed as most wines have prices clustered around the mean while relatively few wines fetch much higher prices. As a result, I choose to apply a logarithmic transform, which helps make the distribution of price more normal. I use the logged price as the target variable in prediction.

### 5. Prediction
In this section, I select relevant features, create training and testing sets, parametrize the machine learning models with k-fold cross-validation, make out-of-sample predictions, and compare performance to a naive model.

I select all of the following features:

- points: The number of points awarded by Wine Enthusiast, on a scale from 1 to 100.
- variety: The type or blend of grapes used in a wine's production.
- description: A sommelier's review. 
- country: The wine's country of origin.
- province: The wine's province or state of origin.
- region_1: The wine's growing region. This is often, but not always, the name of the wine's area of protected origin, if it possesses protected status.
- winery: The wine's producer. 
- designation: The vineyard within the winery where the grapes that made the wine are from
- vintage: The vintage of the wine, ranging from 1990 to 2016.
- snetiment: Sentiment of the sommelier's review, as calculated with Vader, ranging from -1 to 1.
- contains_tannins: Indicator variable equal to 1 if a sommelier review incudes the term "tannins", and 0 otherwise.
- contains_black: Indicator variable equal to 1 if a sommelier review incudes the term "black", and 0 otherwise.
- contains_vineyard: Indicator variable equal to 1 if a sommelier review incudes the term "vineyard", and 0 otherwise.
- contains_licorice: Indicator variable equal to 1 if a sommelier review incudes the term "licorice", and 0 otherwise.
- contains_dark: Indicator variable equal to 1 if a sommelier review incudes the term "dark", and 0 otherwise.
- contains_cherry: Indicator variable equal to 1 if a sommelier review incudes the term "cherry", and 0 otherwise.
- contains_oak: Indicator variable equal to 1 if a sommelier review incudes the term "oak", and 0 otherwise.
- contains_cabernet: Indicator variable equal to 1 if a sommelier review incudes the term "cabernet", and 0 otherwise.
- contains_fruity: Indicator variable equal to 1 if a sommelier review incudes the term "fruity", and 0 otherwise.

I create the training dataset by selecting a random sample of the processed dataset, which includes 80% of observations. The testing dataset comprises the complimentary 20% of wines.

In addition to a naive model, which simply assumes that a wine's price is equal to the average price of the testing set, I train and use a linear regression model as well as 5 machine learning models: lasso regression, K-nearest neighbors, random forest, and gradient-boosted decision trees (XGBoost and LightGBM). 

The machine learning models are parameterized by applying K-fold cross valiation, with 5 folds, on a parameter grid. In this way, I am able to optimize key parameters, including the number of neighbors to use when employing K-nearest neighbors or the number of ___ when employing random forests, for example.

Each model's predictive performance, expressed in terms of mean absolute error (MAE) and mean absolute percentage error (MAPE) is shown below:

| Model | MAE | MAPE (%) | MAPE, Relative to Naive |
| ----- | --- | -------- | ----------------------- |
| Naive | 0.53 | 16.30 | 1 |
| Linear Regression | 0.38 | 11.50 | 0.71 |
| Lasso Regression | 0.38 | 11.69 | 0.72 |
| K-Nearest Neighbors | 0.35 | 10.69 | 0.66 |
| Random Forest | 0.28 | 8.45 | 0.52 |
| XGBoost | 0.26 | 7.81 | 0.48 |
| LightGBM | 0.27 | 8.11 | 0.50 |

The naive model, which makes only unconditional predictions, is found to perform the worst while the gradient-boosted decision trees are found to perform best. In particular, the XGBoost and LightGBM models are found to record MAPEs of 7.81% and 8.11%, respectivey. As such, these models make predictions that are, on average, only 8% away from a wine's actual log(price), making them more than twice as accurate as the naive model. XGBoost remains the best performing model if price, rather than log(price), is used as a target variable. However, the model's MAPE rises to nearly 30% when predicting the untransformed price variable.

In addition, I plot actual and predicted log price values for the 6 machine learning models.

<img src="https://github.com/robertialenti/Wine/raw/main/output/predicted_actual_combined.png" width="500" height="400">

Finally, I examine feature importance. As expected, Wine Enthusiasts' points are generally good predictors of price. A wine's grape country and region of interest, as well as its grape blend are also consistently strong predictors. The sommelier's reviews are comparatively less strong predictors.

Unfortunately, the sommeliers responsible for providing reviews did not also offer price assessments. As such, we cannot compare the performance of the models to the experts' best guess of a wine's price.
