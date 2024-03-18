import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_histograms_for_all_courses(filepath):
    df = pd.read_csv(filepath)
    
    # Identifier les cours en excluant les colonnes non pertinentes
    courses = df.columns.difference(['Index', 'Name', 'First Name', 'Birthday', 'Best Hand', 'Hogwarts House'])
    
    # Pour chaque cours, créer un histogramme des scores par maison
    for course in courses:
        plt.figure(figsize=(10, 6))  # Définit la taille de la figure pour une meilleure lisibilité
        
        # Filtrer les données par maison et créer un histogramme pour chaque maison
        for house in df['Hogwarts House'].unique():
            house_data = df[df['Hogwarts House'] == house][course].dropna()
            plt.hist(house_data, bins=15, alpha=0.5, label=house)
        
        plt.title(f'Histogram of {course} Scores by House')
        plt.xlabel('Scores')
        plt.ylabel('Number of Students')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: histogram.py dataset.csv")
    else:
        filepath = sys.argv[1]
        plot_histograms_for_all_courses(filepath)
