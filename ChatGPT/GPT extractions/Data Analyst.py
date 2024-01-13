import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class DataAnalystGPT:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        """ Load data from a specified path """
        _, file_extension = os.path.splitext(data_path)
        try:
            if file_extension == '.csv':
                return pd.read_csv(data_path)
            elif file_extension in ['.xls', '.xlsx']:
                return pd.read_excel(data_path)
            else:
                print("Unsupported file format.")
                return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def describe_data(self):
        """ Provide a basic description of the data """
        return self.data.describe()

    def visualize_data(self, columns=None):
        """ Create basic visualizations for the data """
        if columns is None:
            columns = self.data.columns

        for column in columns:
            if self.data[column].dtype == 'object':
                sns.countplot(y=column, data=self.data)
                plt.show()
            else:
                self.data[column].hist()
                plt.title(column)
                plt.show()

    def preprocess_data(self):
        """ Basic data preprocessing steps """
        # Handling missing values
        self.data.dropna(inplace=True)

        # Standardizing numeric data
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def correlation_analysis(self):
        """ Perform a correlation analysis """
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()

    def basic_statistical_analysis(self):
        """ Perform basic statistical tests """
        results = {}
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            stat, p = stats.normaltest(self.data[col])
            results[col] = {'statistic': stat, 'p_value': p}
        return results

    def detect_outliers(self, method='IQR'):
        """ Detect outliers in the dataset """
        if method == 'IQR':
            Q1 = self.data.quantile(0.25)
            Q3 = self.data.quantile(0.75)
            IQR = Q3 - Q1
            return ((self.data < (Q1 - 1.5 * IQR)) | (self.data > (Q3 + 1.5 * IQR))).any()

    def simple_linear_regression(self, target_column):
        """ Perform a simple linear regression on the dataset """
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {'model': model, 'mse': mse, 'r2_score': r2}

if __name__ == "__main__":
    data_path = "path_to_your_data.csv" # Replace with your data file path
    analyst = DataAnalystGPT(data_path)

    # Example usage
    print(analyst.describe_data())
    analyst.visualize_data()
    analyst.preprocess_data()
    analyst.correlation_analysis()
    print(analyst.basic_statistical_analysis())
    print(analyst.detect_outliers())
    regression_results = analyst.simple_linear_regression('your_target_column') # Replace 'your_target_column' with the actual column name
    print(regression_results)
