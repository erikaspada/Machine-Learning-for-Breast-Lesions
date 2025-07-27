import matplotlib.pyplot as plt
import seaborn as sns
def plot_class_distribution(df, column_name):
    """
    Plots the distribution of a binary column ('Status' or 'Upgrade') using a bar plot.
    """
    order = [1, 0]
    palette = {1: 'skyblue', 0: 'lightgreen'}

    plt.figure(figsize=(6, 4))
    sns.countplot(x=column_name, data=df, palette=palette, order=order)
    plt.title(f'Distribution of class "{column_name}"')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
