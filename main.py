import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_and_clean_data(url):
    """
    Loads data from URL, filters for specific locations, handles missing values,
    and renames target columns for 'RainToday' prediction.
    """
    print("Loading data...")
    df = pd.read_csv(url)
    df = df.dropna()
    
    # Rename columns to shift prediction target to 'RainToday'
    df = df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})
    
    # Filter for specific region to reduce variance
    df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]
    
    return df

def feature_engineering(df):
    """
    Converts Date to Season and drops the original Date column.
    """
    print("Engineering features...")
    
    def date_to_season(date):
        month = date.month
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        elif month in [9, 10, 11]: return 'Spring'
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Date'].apply(date_to_season)
    df = df.drop(columns=['Date'])
    return df

def build_pipeline(X_train):
    """
    Constructs the preprocessing and classification pipeline.
    """
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Initialize with Random Forest
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline, numeric_features, categorical_features

def evaluate_model(model, X_test, y_test, model_name):
    """
    Prints classification report and displays confusion matrix.
    """
    print(f"\n--- Evaluating {model_name} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

def main():
    # 1. Load Data
    data_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
    df = load_and_clean_data(data_url)
    df = feature_engineering(df)

    # 2. Split Data
    X = df.drop(columns='RainToday', axis=1)
    y = df['RainToday']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3. Build Pipeline
    pipeline, num_feats, cat_feats = build_pipeline(X_train)

    # 4. Model 1: Random Forest
    print("\nTraining Random Forest...")
    param_grid_rf = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid_search = GridSearchCV(pipeline, param_grid_rf, cv=cv, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    evaluate_model(grid_search, X_test, y_test, "Random Forest")

    # 5. Feature Importance (RF only)
    print("\nExtracting Feature Importances...")
    feature_names = num_feats + list(grid_search.best_estimator_['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(cat_feats))
    
    importances = grid_search.best_estimator_['classifier'].feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Important Features (Random Forest)')
    plt.show()

    # 6. Model 2: Logistic Regression
    print("\nTraining Logistic Regression...")
    pipeline.set_params(classifier=LogisticRegression(random_state=42))
    grid_search.estimator = pipeline
    
    param_grid_lr = {
        'classifier__solver': ['liblinear'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__class_weight': [None, 'balanced']
    }
    
    grid_search.param_grid = param_grid_lr
    grid_search.fit(X_train, y_train)
    
    evaluate_model(grid_search, X_test, y_test, "Logistic Regression")

if __name__ == "__main__":
    main()