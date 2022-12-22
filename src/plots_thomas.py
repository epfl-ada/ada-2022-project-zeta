import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_processing import countries
from src.helpers_thomas import *


POLITICS = "History and Society.Politics and government"

N_EUROPE = "Geography.Regions.Europe.Northern Europe"
W_EUROPE = "Geography.Regions.Europe.Western Europe"
S_EUROPE = "Geography.Regions.Europe.Southern Europe"


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
    plt.title("Silhouette Score for Each Value of K")
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

    plt.title("Reduced Mobility vs. Response Time")
    plt.xlabel("Response time (days)")
    plt.ylabel("Reduced mobility (days)")





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

    # keep only rows where date is >  "2020-01-01 and < "2020-04-30"
    df_transport = df_transport.loc[pd.to_datetime("2020-01-01"):pd.to_datetime("2020-04-30")]

    # create a figure of a certain size
    plt.figure(figsize=(7, 4), dpi = 100)
    plt.plot(df_transport)

    plt.ylim(-100, 100)

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

    plt.ylim(-100, 100)

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
    """Plot polling data for a given country. The dataframe `df` is grouped by month, and the mean and standard deviation of each party is plotted."""

    parties = df.columns[1:]
    scores = group_date(df, parties)

    # perform linear interpolation if there are some months with no data
    scores = linear_interpolation(scores, parties)

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

    plt.title(f"Polling Data for {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(idxs[::3], rotation=45)
    plt.ylabel("Percentage")

    plt.axvline(
        x=df_dates.loc[country, "1st death"].strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="1st death",
    )

    plt.legend(loc="lower left")
    plt.show()



def plot_country_alignment(df, country, df_dates):

    # map each party to its political alignment
    df.columns = ["Sample Size"] + list(political_alignment_2[country].values())

    # sum columns that belong to the same alignment
    df = df.groupby(df.columns, axis = 1).sum()

    alignments = df.columns[1:]
    # groupn the dataframe by month to obtain weighted mean and standard deviation of each alignment
    scores = group_date(df, alignments)

    # perform linear interpolation if there are some months with no data
    scores = linear_interpolation(scores, alignments)


    # Plot the data
    idxs = scores.index.map(lambda x: x.strftime("%Y-%m"))

    for alignment in alignments:
        plt.fill_between(
            idxs,
            scores["avg_" + alignment] - scores["std_" + alignment],
            scores["avg_" + alignment] + scores["std_" + alignment],
            alpha=0.2,
        )
        plt.plot(idxs, scores["avg_" + alignment], label=alignment)

    plt.title(f"Political alignment popularity for {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(idxs[::3], rotation=45)
    plt.ylabel("Percentage")

    plt.axvline(
        x=df_dates.loc[country, "1st death"].strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="1st death",
    )

    plt.legend(loc="lower left")
    plt.show()


"""def plot_countries_alignment(dfs, countries, number_countries):

    # merge dfs into one dataframe
    dfs = pd.concat(dfs, axis = 1)

    alignments = []
    for i in number_countries:
        # map each party to its political alignment
        dfs[i].columns = ["Sample Size"] + list(political_alignment[countries[i]].values())

        # get alignments that appears at least once in the data
        for alignment in dfs[i].columns[1:]:
            if alignment not in alignments:
                alignments.append(alignment)
        
        dfs[i] = dfs[i].groupby(dfs[i].columns, axis = 1).sum()

    dfs = pd.concat(dfs, axis = 1)

    print(dfs)

        # sum columns that belong to the same alignment"""
        







def plot_pageviews(df, country, topic, df_dates):
    """Plot pageviews data for a given country and topic."""

    # Select last available year of data
    df_clipped = df.loc[
        pd.date_range(pd.to_datetime("2020-01-01"), pd.to_datetime("2020-04-30"))
    ]

    plt.figure(figsize=(7, 4))

    # Rolling average across 14 days
    plt.plot(df_clipped.index, df_clipped.rolling(14).mean(), label="_nolegend_")

    if topic == POLITICS:
        plt.title(f"Daily Pageviews for Politics in {countries[country]}")
    elif topic == N_EUROPE:
        plt.title(f"Daily Pageviews for Northern Europe in {countries[country]}")
    elif topic == W_EUROPE:
        plt.title(f"Daily Pageviews for Western Europe in {countries[country]}")
    elif topic == S_EUROPE:
        plt.title(f"Daily Pageviews for Southern Europe in {countries[country]}")

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()



    plt.axvline(
        x=pd.to_datetime(df_dates.loc[country, "Mobility"].strftime("%Y-%m-%d")),
        color="red",
        linestyle="--",
    )
    if country != "es":
        plt.axvline(
            x=pd.to_datetime(df_dates.loc[country, "Normalcy"].strftime("%Y-%m-%d")),
            color="green",
            linestyle="--",
        )

    plt.legend(["Abnormal mobility", "Normal mobility"], loc="upper left")

    plt.show()

def plot_pageviews_2(df, country, topic, df_dates):
    """Plot pageviews data for a given country and topic."""

    # Select last available year of data
    df_clipped = df.loc[
        pd.date_range(pd.to_datetime("2020-01-01"), pd.to_datetime("2020-04-30"))
    ]
    # print the index of the dataframe
    plt.figure(figsize=(7, 4), dpi = 100)

    # Rolling average across 14 days
    
    plt.plot(df_clipped.index, df_clipped.rolling(14).mean(), label="_nolegend_")

    if topic == POLITICS:
        plt.title(f"Daily Pageviews for Politics in {countries[country]}")
    elif topic == N_EUROPE:
        plt.title(f"Daily Pageviews for Northern Europe in {countries[country]}")
    elif topic == W_EUROPE:
        plt.title(f"Daily Pageviews for Western Europe in {countries[country]}")
    elif topic == S_EUROPE:
        plt.title(f"Daily Pageviews for Southern Europe in {countries[country]}")

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()

    dates = get_dates(df_dates, country)
    plot_dates(dates)

    plt.show()


    
def plot_mobility_pageviews_covid(df_transport1, df_pageviews, country, topic, df_dates):
    """Plot mobility data for a given country. The data is split into driving and walking data and plotted together."""
    df_drive = df_transport1.loc[country, 'driving']
    df_walk = df_transport1.loc[country, 'walking']
    df_transport = pd.concat([df_drive, df_walk], axis=1)

    # convert indexes to datetime 
    df_transport.index = pd.to_datetime(df_transport.index)

    # keep only rows where date is >  "2020-01-01 and < "2020-04-30"
    df_transport = df_transport.loc[pd.to_datetime("2020-01-01"):pd.to_datetime("2020-04-20")]

    # Select last available year of data
    df_clipped = df_pageviews.loc[pd.date_range(pd.to_datetime("2020-01-01"), pd.to_datetime("2020-04-20"))]

    legends = ['Driving', 'Walking', 'First case', 'First death']
    legends2 = ['First case', 'First death']
    ticks_x = [pd.to_datetime("2020-01-15"), pd.to_datetime("2020-02-01"), pd.to_datetime("2020-02-15"),
    pd.to_datetime("2020-03-01"), pd.to_datetime("2020-03-15"), pd.to_datetime("2020-04-01"), pd.to_datetime("2020-04-15")]

    plt.figure(1, figsize=(5, 7))
    plt.subplot(2, 1, 1)
    plt.plot(df_transport)
    plt.ylim(-100, 50)
    plt.ylabel("Percentage change")
    plt.xlim(pd.to_datetime("2020-01-15"), pd.to_datetime("2020-04-20"))
    dates = get_dates(df_dates, country)
    plot_dates(dates)
    if dates['Lockdown'] is not pd.NaT:
        legends.append('Lockdown')
    plt.title('Percentage of mobility change in {}'.format(country))
    plt.legend(legends)
    #plt.xticks(ticks= [],rotation=45)
    plt.xticks(ticks = ticks_x, labels=[], rotation=45)

    plt.subplot(2, 1, 2)
    # Rolling average across 14 days
    plt.plot(df_clipped.index, df_clipped.rolling(14).mean(), color='black', label="_nolegend_")
    plt.title(f"Daily pageviews for covid related pages {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(ticks = ticks_x, rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()
    plt.xlim(pd.to_datetime("2020-01-15"), pd.to_datetime("2020-04-20"))
    dates = get_dates(df_dates, country)
    plot_dates(dates)
    plt.legend(legends2, loc="upper left")

    plt.show()