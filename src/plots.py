import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_processing import countries
from src.helpers import *


def plot_silhouettes(X, k_min, k_max):
    """Plot Silhouette Scores for a range of clusters between `k_min` and `k_max` inclusive."""

    silhouettes = []

    for k in range(k_min, k_max + 1):
        # Cluster the data and assign labels
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)
        # Compute the mean Silhouette Coefficient over all samples
        score = silhouette_score(X, labels)
        silhouettes.append({"k": k, "score": score})

    # Convert to dataframe
    silhouettes = pd.DataFrame(silhouettes)

    # Plot the data
    plt.plot(silhouettes.k, silhouettes.score)
    plt.title("Silhouette Score for each value of K")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")


def plot_mobility_response(df, labels=None):
    """Plot response time and duration of reduced mobility period of countries. Optionally group the countries by `labels`."""

    ax = sns.scatterplot(
        x=df["Response time"],
        y=df["Reduced mobility"],
        hue=labels + 1,
        palette=sns.color_palette(n_colors=len(set(labels))),
    )

    for idx, row in df.iterrows():
        ax.text(row["Response time"] + 0.5, row["Reduced mobility"], idx)

    plt.title("Reduced mobility vs. Response time")
    plt.xlabel("Response time (days)")
    plt.ylabel("Reduced mobility (days)")


def plot_polling_data(df, country, df_dates):
    """Plot polling data for a given country. The dataframe df is grouped by month and the mean and standard deviation of each party is plotted."""

    df_grouped = df.groupby("Date")

    parties = df.columns[1:]

    # Create dataframe with average and standard deviation for each party
    avg = df_grouped.apply(
        lambda x: pd.Series(
            {
                "avg_" + party: calculate_mean(x[party], x["Sample Size"])
                for party in parties
            }
        )
    )
    std = df_grouped.apply(
        lambda x: pd.Series(
            {
                "std_" + party: calculate_std(x[party], x["Sample Size"])
                for party in parties
            }
        )
    )

    scores = pd.concat([avg, std], axis=1)
    idxs = scores.index.map(lambda x: x.strftime("%Y-%m"))

    # Plot data
    for party in parties:
        plt.fill_between(
            idxs,
            scores["avg_" + party] - scores["std_" + party],
            scores["avg_" + party] + scores["std_" + party],
            alpha=0.2,
        )
        plt.plot(idxs, scores["avg_" + party], label=party)

    plt.title(f"Polling data for {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(idxs[::3], rotation=45)
    plt.ylabel("Percentage")
    plt.legend()

    plt.axvline(x=df_dates.loc[country, "1st death"].strftime("%Y-%m"), color="black")

    plt.show()
