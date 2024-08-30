#%% Section 1: Preliminaries
# General
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import warnings
import json

# Figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18
plt.rc("savefig", dpi = 300)
sns.set_theme(style = "ticks", font_scale = 1, font = "DejaVu Sans")

# Natural Language Processing
from textblob import TextBlob

# Machine Learning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Other
warnings.filterwarnings("ignore", category=FutureWarning, message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
pd.set_option("display.expand_frame_repr", False)

# Paths
filepath = "C:/Users/rialenti/OneDrive - Harvard Business School/Desktop/Work/Other/Wine/"


#%% Section 2: Importing and Preparing Data
# Import Data
df = pd.read_csv('data/data.csv', usecols=lambda column: not column.startswith('Unnamed'))

# Remove Duplicate Reviews
df = df.drop_duplicates(subset=['description'])

# Drop Unwanted Variables
df = df.drop(['taster_twitter_handle', 'taster_name'], axis=1)

# Create Vintage Variable
df['vintage'] = df['title'].str.extract(r'(\b\d{4}\b)', expand=False)
df['vintage'] = pd.to_numeric(df['vintage'], errors='coerce')
df = df.dropna(subset=['vintage'])
df["vintage"] = df["vintage"].astype(int)

# Remove Wines with Implausible Price
df = df[df["price"] > 0]

# Remove Poorly Populated Vintages
df = df[df["vintage"] >= 1990]
df = df[df["vintage"] <= 2016]

# Convert String Variables to Categorical Variable
for variable in ["country", "province", "region_1", "region_2", "winery", "designation", "variety"]:
    df[variable] = pd.Categorical(df[variable])
    
    
#%% Section 4: Natural Language Processing
# Option 1
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

tqdm.pandas()
df['sentiment'] = df['description'].progress_apply(analyze_sentiment)


# Option 2
from transformers import pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment_transformers(text):
    result = sentiment_analyzer(text)[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

df['sentiment'] = df['description'].progress_apply(analyze_sentiment_transformers)


#%% Exploratory Analysis
# 1. Variety Distribution
df_plot = df
variety_counts = df_plot['variety'].value_counts(dropna=False)
df_plot['variety_grouped'] = df_plot['variety'].apply(lambda x: x if variety_counts[x] >= 1000 else 'Other')
final_counts = df_plot['variety_grouped'].value_counts().reset_index()
final_counts.columns = ['variety', 'count']
final_counts = final_counts.sort_values(by = "count", ascending = False)
final_counts = pd.concat([
    final_counts[final_counts['variety'] != 'Other'],
    final_counts[final_counts['variety'] == 'Other']
])
final_counts = final_counts.reset_index(drop=True)

ax = sns.barplot(data = final_counts, x = 'count', y = 'variety')
plt.grid(False)
plt.ylabel("Grape Variety")
plt.xlabel("Frequency")
plt.title("Number of Wines by Grape Variety")
plt.savefig(filepath + "figures/variety_distribution.png", bbox_inches = "tight")
plt.show()

# 2. Country Distribution
df_plot = df
country_counts = df_plot['country'].value_counts(dropna=False)
df_plot['country_grouped'] = df_plot['country'].apply(lambda x: x if country_counts[x] >= 100 else 'Other')
final_counts = df_plot['country_grouped'].value_counts().reset_index()
final_counts.columns = ['country', 'count']
final_counts = final_counts.sort_values(by = "country")
final_counts = final_counts.sort_values(by = "count", ascending = False)
final_counts = pd.concat([
    final_counts[final_counts['country'] != 'Other'],
    final_counts[final_counts['country'] == 'Other']
])
final_counts = final_counts.reset_index(drop=True)

ax = sns.barplot(data = final_counts, x = 'count', y = 'country')
plt.grid(False)
plt.xticks(rotation=90)
plt.ylabel("Country")
plt.xlabel("Frequency")
plt.title("Country Distribution")
plt.savefig(filepath + "figures/country_distribution.png", bbox_inches = "tight")
plt.show()

# 3. Vintage Distribution
df_plot = df
vintage_counts = df_plot['vintage'].value_counts().reset_index()
vintage_counts.columns = ['vintage', 'count']
vintage_counts = vintage_counts.sort_values(by = "vintage")

ax = sns.barplot(data = vintage_counts, x = 'vintage', y = 'count')
plt.grid(False)
plt.xticks(rotation=90)
plt.ylabel("Frequency")
plt.xlabel("Vintage")
plt.title("Vintage Distribution")
plt.savefig(filepath + "figures/vintage_distribution.png", bbox_inches = "tight")
plt.show()

# 4. Price Distribution
df_plot = df
df_plot["log_price"] = np.log(df_plot["price"])
plt.hist(df_plot['log_price'], bins=10, edgecolor='black')
plt.grid(False)
plt.ylabel("Frequency")
plt.xlabel("Log(Price)")
plt.title("Price Distribution")
plt.savefig(filepath + "figures/price_distribution.png", bbox_inches = "tight")
plt.show()

# 5. Relationship Between Price and Points
df_plot = df
df_plot["vintage"] = df_plot["vintage"].astype(int)
sns.scatterplot(
    data=df_plot, 
    x='points', 
    y='price', 
    hue='vintage', 
    palette='viridis',
    legend=None
)

norm = plt.Normalize(df_plot['vintage'].min(), df_plot['vintage'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
plt.xlabel('Points')
plt.ylabel('Price ($)')
plt.title('Price vs Points, Colored by Vintage')
cbar = plt.colorbar(sm)
cbar.set_label('Vintage', labelpad=10, fontsize=12)
cbar.ax.set_title('Vintage', pad=15, fontsize=12)
cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.set_ylabel('', rotation=0)
plt.savefig(filepath + "figures/price_points_vintage.png", bbox_inches = "tight")
plt.show()

# 6. Relationship Between Price and Points
df_plot = df
sns.scatterplot(
    data=df_plot, 
    x='points', 
    y='price', 
    hue='sentiment', 
    palette='viridis',
    legend=None
)

norm = plt.Normalize(df_plot['sentiment'].min(), df_plot['sentiment'].max())
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
plt.xlabel('Points')
plt.ylabel('Price ($)')
plt.title('Price vs Points, Colored by Sentiment')
cbar = plt.colorbar(sm)
cbar.set_label('Sentiment', labelpad=10, fontsize=12)
cbar.ax.set_title('Sentiment', pad=15, fontsize=12)
cbar.ax.yaxis.set_label_position('right')
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.set_ylabel('', rotation=0)
plt.savefig(filepath + "figures/price_points_sentiment.png", bbox_inches = "tight")
plt.show()


df_plot = df
corr_matrix = df_plot[["price", "points", "vintage", "sentiment"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#%% Section 5: Make and Evaluate Predictions
# Define Function for Selecting Relevant Features
def select_features(data):
    data = data[["points", "vintage", "sentiment", "designation", "country", "province", "region_1", "winery", "variety", "price"]]
    data = data.dropna()
    return data


# Define Function for Parametrizing Models
def parametrize_models(X_train, y_train):
    # Define Parameter Grid
    param_grids = {
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7, 9]
        },
        "Lasso": {
            "alpha": [0.01, 0.1, 1.0, 10.0]
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7, -1]
        }
    }

    # Specify Models Not Needing Parametrization
    models = {
        "Linear Regression": LinearRegression()
    }
    
    # Perform Cross-Validation to Optimize Parameters for All Other Models
    for model_name, model in [
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("Random Forest", RandomForestRegressor(random_state = 1)),
        ("XGBoost", XGBRegressor(random_state = 1)),
        ("Lasso", Lasso(max_iter = 10000)),
        ("LightGBM", LGBMRegressor(random_state = 1)),
    ]:
        grid_search = GridSearchCV(model, 
                                   param_grids[model_name], 
                                   cv=5, 
                                   scoring='neg_mean_absolute_error', 
                                   n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        models[model_name] = best_model
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    
    # Return Parametrized Models
    return models
        

# Define Funtion for Making Predictions
def make_predictions(data, target, test_size):
    # Time for Prediction
    start_time = time.time()
    
    # Select Features
    data = select_features(data)
    
    # Encode Categorical Variables
    for col in data.select_dtypes(include=['category']).columns:
        data[col] = data[col].cat.codes
    
    # Create Training and Testing Subsamples
    split_index = int(len(data) * (1 - test_size))
    data = data.sample(frac = 1, random_state = 1).reset_index(drop=True)
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    
    X_train, y_train = train.drop(target, axis = 1), train[target]
    X_test, y_test = test.drop(target, axis = 1), test[target]
    
    # Parametrize Models
    models = parametrize_models(X_train, y_train)
    
    # Create Empty List to Hold Evaluation Results
    results = []
    
    # Define Empty Dictionary for Feature Importance
    feature_importances = {}
    
    # Train, Test, and Evaluate Each Machine Learning Model
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": model_name,
            "MAE": mae, 
            "MAPE": mape,
            "R2": r2
            })
        
        # Collect Feature Importance for Models That Provide Them
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = model.feature_importances_
            
        # Plot Predicted vs Actual Prices
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Predicted vs Actual Prices for {model_name}')
        plt.savefig(filepath + f"output/predicted_actual_{model_name}.png", bbox_inches = "tight")
        plt.show()
        
    # Test and Evaluate Rule-of-Thumb Model
    thumb_predictions = test.copy()
    thumb_predictions['predicted_price'] = thumb_predictions.groupby(['points'])['price'].transform('mean')
    
    mae = mean_absolute_error(y_test, thumb_predictions['predicted_price'])
    mape = mean_absolute_percentage_error(y_test, thumb_predictions['predicted_price'])*100
    r2 = r2_score(y_test, thumb_predictions['predicted_price'])
    
    results.append({
        "Model": "Rule-of-Thumb",
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    })
        
    
    # Plot Feature Importances
    if feature_importances:
        for model_name, importances in feature_importances.items():
            sorted_idx = importances.argsort()
            plt.barh(X_train.columns[sorted_idx], importances[sorted_idx])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importances for {model_name}')
            plt.savefig(filepath + f"output/feature_importance_{model_name}.png", bbox_inches = "tight")
            plt.show()
            
    # Save Model Performance
    with open(filepath + 'output/model_performance.json', 'w') as f:
        json.dump(results, f, indent = 4)
    
    # Return Models and Results
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Model parametrization, training, testing, and evaluation was completed in{execution_time: .2f} minutes.")
    return models, results


# Make Predictions
models, results = make_predictions(df, "price", 0.20)

