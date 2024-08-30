# Predicting Wine Prices
This project aims to predict prices for a selection of wines reviewed by sommeliers. I employ a number of commonly used machine learning techniques.

## Data
The data was scraped by [Zach Thoutt](https://github.com/zackthoutt/wine-deep-learning) in 2017 from [Wine Enthusiast](https://www.wineenthusiast.com/?s=&search_type=shop). The raw dataset includes nearly 150,000 reviews. For each wine, we have information about its production, its country and region of origin, the score assigned to it by Wine Enthusiast, as well as its price in US dollars.

## Code
The code is separated into 5 sections.

### 1. Preliminaries
I begin by importing widely used libraries for data analysis, machine learning, and natural language processing. I extract the vintage year, which I believe to be an important predictor of a wine's value, from the `descriptor` variable.

### 2. Importing and Cleaning Data
I read in the data, __. I remove variables I deem to have little predictive power, namely `Taster Name` and `Taster Twitter Handle`. 

The processed dataset includes xx observations, with the following variables:

- Points: the number of points WineEnthusiast rated the wine on a scale of 1-100 (though they say they only post reviews for wines that score >=80)
- Title: the title of the wine review, which often contains the vintage if you're interested in extracting that feature
- Variety: the type of grapes used to make the wine (ie Pinot Noir)
- Description: a few sentences from a sommelier describing the wine's taste, smell, look, feel, etc.
- Country: the country that the wine is from
- Province: the province or state that the wine is from
- Region 1: the wine growing area in a province or state (ie Napa)
- Region 2: sometimes there are more specific regions specified within a wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank
- Winery: the winery that made the wine
- Designation: the vineyard within the winery where the grapes that made the wine are from
- Price: the cost for a bottle of the wine
- Taster Name: name of the person who tasted and reviewed the wine
- Taster Twitter Handle: Twitter handle for the person who tasted ane reviewed the wine

### 3. Natural Language Processing
I evaluate the sentiment of each description using a transformer, which is a deep learning model better suited to interpret human language.

### 4. Exploratory Analysis
Before proceeding to making predictions, I visualize some of the data's features to get a better sense of how variables are distributed and related.

### 5. Prediction
In this section, I select relevant features, parametrize the machine learning models with k-fold cross-validation, make out-of-sample predictions, and compare performance in comparison to a naive rule-of-thumb model.

The features selected include all of the numerical variables, namely ___, as well as categorical variables including ____.

I create the training dataset by selecting a random sample of the processed dataset with ___% of the ___. The testing dataset comprises the complimentary 20% of wines.

The rule-of-thumb model predicts a wine's price.
