import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

def create_pair_plot(filepath):
    df = pd.read_csv(filepath)
    
    # Sélectionner uniquement les colonnes des matières
    subjects = ['Herbology', 'Defense Against the Dark Arts','Ancient Runes', 'Potions' ,'Charms']
    
    # Ajouter 'Hogwarts House' à la liste pour utiliser comme couleur dans le pair plot
    columns_to_keep = subjects + ['Hogwarts House']
    
    # Filtrer le DataFrame pour ne garder que les colonnes sélectionnées
    df_filtered = df[columns_to_keep]
    
    # Créer un pair plot avec seaborn
    sns.pairplot(df_filtered, hue='Hogwarts House', diag_kind='kde', corner=True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <path_to_your_dataset>")
        sys.exit(1)

    filepath = sys.argv[1]
    create_pair_plot(filepath)
