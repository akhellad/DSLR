import pandas as pd
import numpy as np
import sys

def my_count(values):
    count = 0
    for value in values:
        if not np.isnan(value):
            count += 1
    return count

def my_mean(values):
    total_sum = 0
    count = my_count(values)
    for value in values:
        if not np.isnan(value):
            total_sum += value
    return total_sum / count if count else 0

def my_std(values, mean):
    count = my_count(values) - 1
    variance_sum = sum((x - mean) ** 2 for x in values if not np.isnan(x))
    return (variance_sum / count) ** 0.5 if count else 0

def my_min(values):
    min_value = np.inf
    for value in values:
        if not np.isnan(value) and value < min_value:
            min_value = value
    return min_value if min_value != np.inf else 0

def my_max(values):
    max_value = -np.inf
    for value in values:
        if not np.isnan(value) and value > max_value:
            max_value = value
    return max_value if max_value != -np.inf else 0

def my_percentile(values, percentile):
    sorted_values = sorted(v for v in values if not np.isnan(v))
    index = (len(sorted_values) - 1) * percentile / 100
    if index % 1 == 0:
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index % 1)

def calculate_statistics(df):
    stats = {
        'Feature': [],
        'Count': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        '25%': [],
        '50%': [],
        '75%': [],
        'Max': []
    }

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        values = df[column].dropna()
        stats['Feature'].append(column)
        stats['Count'].append(my_count(values))
        stats['Mean'].append(my_mean(values))
        stats['Std'].append(my_std(values, my_mean(values)))
        stats['Min'].append(my_min(values))
        stats['25%'].append(my_percentile(values, 25))
        stats['50%'].append(my_percentile(values, 50))
        stats['75%'].append(my_percentile(values, 75))
        stats['Max'].append(my_max(values))
    
    return pd.DataFrame(stats)

def describe_dataset(filepath):
    df = pd.read_csv(filepath)  
    description = calculate_statistics(df)
    description_transposed = description.transpose()
    description_transposed.columns = description_transposed.iloc[0]
    description_transposed = description_transposed.drop(description_transposed.index[0])
    
    print(description_transposed)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: describe.py dataset.csv")
    else:
        filepath = sys.argv[1]
        describe_dataset(filepath)
