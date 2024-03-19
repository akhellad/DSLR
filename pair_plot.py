import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def create_pair_plot(filepath):
    df = pd.read_csv(filepath)
    
    # Optionnel : Sélectionner un sous-ensemble de caractéristiques pour le pair plot
    # Cela peut être utile si tu as beaucoup de caractéristiques et que tu veux te concentrer sur quelques-unes d'entre elles
    features = ['Feature1', 'Feature2', 'Feature3', 'Hogwarts House']
    df = df[features]
    
    # Créer un pair plot. Si tu as une colonne 'Hogwarts House' ou toute autre catégorie, tu peux l'utiliser comme hue pour colorer les points.
    # Remplace 'Hogwarts House' par le nom de ta colonne de catégories si nécessaire
    sns.pairplot(df, hue='Hogwarts House', diag_kind='kde', corner=True)
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pair_plot.py dataset.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    create_pair_plot(filepath)