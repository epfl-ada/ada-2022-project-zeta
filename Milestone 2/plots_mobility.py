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


# create optional argument for lockdown

def plot_dates(dates):
    """Plot dates of first case, first death and lockdown."""
    first_case, first_death, lockdown = dates.values
    plt.axvline(x=first_case, color="green", linestyle="--")
    plt.axvline(x=first_death, color="red", linestyle="--")

    # some countries do not have lockdown dates
    if lockdown is not pd.NaT:
        plt.axvline(x=lockdown, color="black", linestyle="--")


    

def plot_driving_walking_mobility(df, df_dates, country):
    """Plot mobility data for a given country. The data is split into driving and walking data and plotted together."""

    df_drive = df.loc[country, 'driving']
    df_walk = df.loc[country, 'walking']
    df_transport = pd.concat([df_drive, df_walk], axis=1)

    # convert indexes to datetime
    df_transport.index = pd.to_datetime(df_transport.index)
    plt.plot(df_transport)


    legends = ['Driving', 'Walking', 'First case', 'First death']

    dates = get_dates(df_dates, country)
    plot_dates(dates)

    if dates['Lockdown'] is not pd.NaT:
        legends.append('Lockdown')

    plt.title('Percentage of mobility change in {}'.format(country))
    plt.xlabel('Date')
    plt.legend(legends)
    plt.xticks(rotation=45)
    plt.show()



def plot_google_mobility(df_mobility, df_dates, columns_dict, title, country):
    """Plot Google mobility data for a given country. The title arguement give the column we want to plot."""

    ti = columns_dict[title]
    df = df_mobility.loc[country][ti]
    plt.plot(df)

    legends = ['Pageviews', 'First case', 'First death']

    dates = get_dates(df_dates, country)
    plot_dates(dates)

    if dates['Lockdown'] is not pd.NaT:
        legends.append('Lockdown')
    

  
    plt.title(f"{title} for {country}")
    plt.xlabel("Date")
    plt.ylabel(f"{ti}")
    plt.legend(legends)
    plt.xticks(rotation=45)

    plt.show()





def plot_polling_data(df, country, df_dates):
    """Plot polling data for a given country. The dataframe df is grouped by month and the mean and standard deviation of each party is plotted."""

    df_grouped = df.groupby("Date")

    # Retrieve parties of the country
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

    # Convert indexes to new wanted displayed format
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
