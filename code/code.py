#%% Section 1: Preliminaries
# General
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os
import warnings

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
from PIL import Image

# Text Analytics
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Machine Learning
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Other
warnings.filterwarnings("ignore", category=FutureWarning, message="The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.")
pd.set_option("display.expand_frame_repr", False)

# Paths
filepath = "C:/Users/rialenti/OneDrive - Harvard Business School/Desktop/Work/Other/Wine/"


#%% Section 2: Importing and Preparing Data
# Import Data
df = pd.read_csv(filepath + 'data/data.csv', usecols=lambda column: not column.startswith('Unnamed'))

# Define Function for Cleaning Data
def clean_data(data):
    # Remove Duplicate Reviews
    data = data.drop_duplicates(subset=['description'])
    
    # Drop Unwanted Variables
    data = data.drop(['taster_twitter_handle', 'taster_name', 'region_2'], axis=1)
    
    # Remove Wines with Implausible Price
    data = data[data["price"] > 0]
    
    # Create Log(Price) Variable
    data["log_price"] = np.log(data["price"])
    
    # Create Vintage Variable
    data['vintage'] = data['title'].str.extract(r'(\b\d{4}\b)', expand=False)
    data['vintage'] = pd.to_numeric(data['vintage'], errors='coerce')
    data = data.dropna(subset=['vintage'])
    data["vintage"] = data["vintage"].astype(int)
    
    # Remove Poorly Populated Vintages
    data = data[data["vintage"] >= 1990]
    data = data[data["vintage"] <= 2016]
    
    # Replace Missing Region With Modal Region for Same Designation
    def replace_with_mode(data, group_col, target_col):
        data[group_col].fillna('Unknown', inplace=True)
        
        # Calculate mode of target_col for each group in group_col
        mode_region = data.groupby(group_col)[target_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        
        # Function to replace NaNs in target_col with mode of group
        def fill_na(row):
            if pd.isna(row[target_col]):
                return mode_region.get(row[group_col], row[target_col])
            return row[target_col]
        
        # Apply the function to replace NaNs
        data[target_col] = data.apply(fill_na, axis=1)
    
    replace_with_mode(data, 'designation', 'region_1')
    
    # Return Clean Data
    return data


df = clean_data(df)
    
    
#%% Section 4: Text Analysis
# Define Function for Extracting Sentiment from Reviews
def get_sentiment_vader(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']


# Assuming df is your DataFrame with wine reviews
tqdm.pandas()
df['sentiment'] = df['description'].progress_apply(get_sentiment_vader)

# Define Function for Extracting Terms with Most Predictive Power
def find_top_correlated_words(df, text_column, target_column, top_n_words, top_n_correlated):
    # Create List of Stop Words
    stop_words = set(stopwords.words('english'))
    
    # Preprocess Text and Exclude Stop Words
    df[text_column] = df[text_column].str.lower().str.replace('[^\w\s]', '', regex=True)
    df[text_column] = df[text_column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Detect Most Common Terms
    all_words = ' '.join(df[text_column].dropna()).split()
    common_words = [word for word, count in Counter(all_words).most_common(top_n_words)]

    # Create a dictionary to store temporary columns
    temp_columns = {}

    # Correlate Each Indicator Variable for Each Term (store in temp_columns)
    correlations = {}
    for word in common_words:
        temp_columns[f'contains_{word}'] = df[text_column].apply(lambda x: 1 if word in x.split() else 0)
        correlations[word] = pd.Series(temp_columns[f'contains_{word}']).corr(df[target_column])

    # Select Terms with Highest Correlation with Price
    sorted_correlations = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    top_correlated_words = sorted_correlations[:top_n_correlated]

    # Convert List of Common Terms to DataFrame
    top_correlated_df = pd.DataFrame(top_correlated_words, columns=['Word', 'Correlation'])

    # Only keep columns for the top correlated words
    top_columns = {f'contains_{word}': temp_columns[f'contains_{word}'] for word, _ in top_correlated_words}

    # Concatenate these columns to the original DataFrame
    df = pd.concat([df, pd.DataFrame(top_columns)], axis=1)

    # Return the DataFrame and the top correlated words
    return top_correlated_df, df


# Usage
top_words_df, df = find_top_correlated_words(df, 
                                             text_column='description', 
                                             target_column='price', 
                                             top_n_words=100, 
                                             top_n_correlated=10)


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
plt.ylabel("")
plt.xlabel("Frequency")
plt.title("Variety Disribution")
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
plt.ylabel("")
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

# 4. Relationship Between Price and Points
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

# Correlation Matrix
df_plot = df
corr_matrix = df_plot.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


#%% Section 5: Make and Evaluate Predictions
# Define Function for Selecting Relevant Features
def select_features(data):
    # Specify Relevant Features to Include
    cols_keep = ["points", "vintage", "sentiment", "designation", "country", "province", "region_1", "winery", "variety", "log_price"] + \
           [col for col in data.columns if "contains" in col]

    # Select Relevant Features
    data = data[cols_keep]
    
    # Drop Observations with Missing Values
    data = data.dropna()
    
    # Encode Categorical Variables
    cols_categoical = ["designation", "country", "province", "region_1", "winery", "variety"]
    for col in cols_categoical:
        data[col] = pd.Categorical(data[col])
        data[col] = data[col].cat.codes
    
    # Return Data with Relevant Features
    print("Features selected.")
    return data


# Define Function for Creating Training and Testing Sets
def create_train_test_sets(data, target_column, test_size):
    # Create Training and Testing Subsamples
    split_index = int(len(data) * (1 - test_size))
    data = data.sample(frac = 1, random_state = 1).reset_index(drop=True)
    
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]
    
    X_train, y_train = train.drop(target_column, axis = 1), train[target_column]
    X_test, y_test = test.drop(target_column, axis = 1), test[target_column]
    
    # Return Training and Testing Sets
    print("Training and test sets created.")
    return train, test, X_train, y_train, X_test, y_test


# Define Function for Parameterizing Models
def parameterize_models(X_train, y_train):
    # Define Parameter Grid
    param_grids = {
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9, 12]
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

    # Specify Models Not Needing Parameterization
    models = {
        "Linear Regression": LinearRegression()
    }
    
    # Perform Cross-Validation to Optimize Parameters for All Other Models
    for model_name, model in [
        ("K-Nearest Neighbors", KNeighborsRegressor()),
        ("Random Forest", RandomForestRegressor(random_state = 1)),
        ("XGBoost", XGBRegressor(random_state = 1)),
        ("Lasso", Lasso(max_iter = 10000)),
        ("LightGBM", LGBMRegressor(random_state = 1, force_col_wise = True)),
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
    
    # Return Parameterized Models
    print("Models parameterized.")
    return models


# Define Function for Training, Testing, and Evaluating Models
def train_test_evaluate(models, train, test, X_train, y_train, X_test, y_test): 
    # Create Empty List to Hold Model Performance Results
    results = []
    
    # Create Empty List to Hold Predicted vs. Actual Plots
    plots = []
    
    # Define Empty Dictionary for Feature Importance
    feature_importances = {}
    
    # Train, Test, and Evaluate Each Machine Learning Model
    for model_name, model in models.items():
        # Fit Model
        model.fit(X_train, y_train)
        
        # Make Prediction
        y_pred = model.predict(X_test)
        
        # Calculate Performance
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        
        results.append({
            "Model": model_name,
            "MAE": mae, 
            "MAPE": mape
            })
        
        # Collect Feature Importance for Models That Provide Them
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = model.feature_importances_
            
        # Plot Predicted vs Actual Prices
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Log(Actual Price)')
        plt.ylabel('Log(Predicted Price)')
        plt.title(f'Predicted vs Actual Prices for {model_name}')
        plot_path = filepath + f"output/predicted_actual_{model_name}.png"
        plt.savefig(plot_path, bbox_inches = "tight")
        plot = Image.open(plot_path)
        plots.append(plot)
        plt.show()
        
    # Test and Evaluate Naive Model
    naive_pred = test.copy()
    naive_pred['predicted_price'] = naive_pred['log_price'].mean()
    
    mae = mean_absolute_error(y_test, naive_pred['predicted_price'])
    mape = mean_absolute_percentage_error(y_test, naive_pred['predicted_price'])*100
    
    results.append({
        "Model": "Naive",
        "MAE": mae,
        "MAPE": mape
    })
    
    plt.scatter(y_test, naive_pred['predicted_price'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Log(Actual Price)')
    plt.ylabel('Log(Predicted Price)')
    plt.title('Predicted vs Actual Prices for Naive Model')
    plot_path = filepath + "output/predicted_actual_naive.png"
    plt.savefig(plot_path, bbox_inches = "tight")
    #plot = Image.open(plot_path)
    #plots.append(plot)
    plt.show()
    
    # Plot Feature Importance by Model
    if feature_importances:
        for model_name, importances in feature_importances.items():
            sorted_idx = importances.argsort()
            plt.barh(X_train.columns[sorted_idx], importances[sorted_idx])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importances for {model_name}')
            plt.savefig(filepath + f"output/feature_importance_{model_name}.png", bbox_inches = "tight")
            plt.show()
    
    # Return Results and Plots
    print("Models trained, tested, and evaluated.")
    return results, plots


# Define Function for Combining Predicted vs. Actual Plots
def combine_plots(plots):
    num_plots = len(plots)
    if num_plots != 6:
        raise ValueError("This function requires exactly 6 plots.")

    # Calculate the dimensions of the combined image
    max_width = max(plot.size[0] for plot in plots)
    max_height = max(plot.size[1] for plot in plots)
    
    # Two rows, three images per row
    combined_width = max_width * 3
    combined_height = max_height * 2

    # Create a new blank image with the combined dimensions
    combined_plot = Image.new('RGB', (combined_width, combined_height))

    # Paste images into the combined image
    for i, plot in enumerate(plots):
        row = i // 3  # 0 for first row, 1 for second row
        col = i % 3   # 0 for first column, 1 for second column, 2 for third column
        x_position = col * max_width
        y_position = row * max_height
        combined_plot.paste(im=plot, box=(x_position, y_position))

    # Save the combined image
    save_path = os.path.join(filepath, 'output/predicted_actual_combined.png')
    combined_plot.save(save_path)
    

# Define Function for Making Predictions
def make_predictions(data):
    # Begin Timer
    start_time = time.time()
    
    # Select Features
    data = select_features(data)
    
    # Create Training and Testing Sets
    train, test, X_train, y_train, X_test, y_test = create_train_test_sets(data, 
                                                                           target_column = "log_price", 
                                                                           test_size = 0.20)
    
    # Parameterize Models
    #models = parameterize_models(X_train, 
    #                             y_train)
    
    # Obtain Results
    results, plots = train_test_evaluate(models, 
                                  train, 
                                  test, 
                                  X_train, 
                                  y_train, 
                                  X_test, 
                                  y_test)
    
    # Combine Predicted vs. Actual Plots
    combine_plots(plots)
    
    # End Timer
    end_time = time.time()
    execution_time = (end_time - start_time)/60
    print(f"Feature selection, model parameterization, training, testing, evaluation, and visualization were completed in{execution_time: .2f} minutes.")
    
    # Return Parameterized Models and Results
    return models, results

# Make Predictions
models, results = make_predictions(df)
