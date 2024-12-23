import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kvas(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x="KVA_Id", data=df, order=df["KVA_Id"].value_counts().index)
    plt.xticks(rotation=90)
    plt.title("Count of Each KVA ID")
    plt.xlabel("KVA ID")
    plt.ylabel("Count")
    plt.show()
