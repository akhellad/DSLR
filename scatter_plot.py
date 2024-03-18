import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def find_most_similar_features(df):
    corr_matrix = df.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    feature_a, feature_b = np.unravel_index(np.argmax(corr_matrix.values), corr_matrix.shape)
    return corr_matrix.columns[feature_a], corr_matrix.columns[feature_b]

def plot_scatter_plot(filepath):
    df = pd.read_csv(filepath)
    df_numeric = df.select_dtypes(include=[np.number])
    
    feature_x, feature_y = find_most_similar_features(df_numeric)
    
    plt.scatter(df[feature_x], df[feature_y])
    plt.title(f'Scatter Plot between {feature_x} and {feature_y}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: scatter_plot.py dataset.csv")
    else:
        filepath = sys.argv[1]
        plot_scatter_plot(filepath)
