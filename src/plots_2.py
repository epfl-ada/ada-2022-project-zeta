import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_processing import countries, languages
from src.helpers_2 import *


POLITICS = "History and Society.Politics and government"

N_EUROPE = "Geography.Regions.Europe.Northern Europe"
W_EUROPE = "Geography.Regions.Europe.Western Europe"
S_EUROPE = "Geography.Regions.Europe.Southern Europe"


def plot_silhouettes(X, k_min, k_max):
    """Plot Silhouette Scores for a range of clusters between `k_min` and `k_max` inclusive.
    
    Args:
        X: dataframe containing the data.
        k_min: minimum number of clusters.
        k_max: maximum number of clusters.
        """

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


def plot_dates(dates):
    """Plot dates of first case, first death and lockdown.
    
    Args: 
        dates: dataframe containing dates of first case, first death and lockdown.
    """

    # retrive the dates
    first_case, first_death, lockdown = dates.values

    plt.axvline(x=first_case, color="green", linestyle="--")
    plt.axvline(x=first_death, color="red", linestyle="--")

    # Some countries do not have lockdown dates
    if lockdown is not pd.NaT:
        plt.axvline(x=lockdown, color="black", linestyle="--")


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


def plot_polling_data(df, country, df_dates):
    """Plot polling data for a given country. The dataframe `df` is grouped by month, and the mean and standard deviation of each party is plotted.
    
    Args:
        df: dataframe containing polling data.
        country: Code of the country.
        df_dates: dataframe containing dates of first case, first death and lockdown."""

    # retrieve parties of the given country
    parties = df.columns[1:]

    # group the dataframe by month to obtain weighted mean and standard deviation of each alignment
    scores = group_date(df, parties)

    # Perform linear interpolation if there are some months with no data
    scores = linear_interpolation(scores, parties)

    # set the indexes of the plot
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

    # plot the vertical line representing the date of the first death
    plt.axvline(
        x=df_dates.loc[country, "1st death"].strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="1st death",
    )

    plt.legend(loc="lower left")
    plt.show()


def plot_pageviews(df, country, topic, df_dates):
    """Plot pageviews data for a given country and topic.
    
    Args:
        df: dataframe containing pageviews data.
        country: Code of the country.
        topic: Topic of the pageviews.
        df_dates: dataframe containing dates of first case, first death and lockdown."""

    # Select last available year of data
    df = df.loc[
        pd.date_range(pd.to_datetime("2019-08-01"), pd.to_datetime("2020-07-31"))
    ]

    # Rolling average across 14 days
    plt.plot(df.rolling(14).mean(), label="_nolegend_")

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

    # plot the vertical line representing the date of the start of abnormal mobility
    plt.axvline(
        x=pd.to_datetime(df_dates.loc[country, "Mobility"].strftime("%Y-%m-%d")),
        color="red",
        linestyle="--",
    )

    # plot the vertical line representing the date of return to normal mobility
    if country != "es":
        plt.axvline(
            x=pd.to_datetime(df_dates.loc[country, "Normalcy"].strftime("%Y-%m-%d")),
            color="green",
            linestyle="--",
        )

    plt.legend(["Abnormal mobility", "Normal mobility"], loc="upper left")

    plt.show()


def plot_pageviews2(df_pageviews, country, df_dates):
    """Plot pageviews data for COVID and politics related pages for a given country.
    
    Args:
        df_pageviews: dataframe containing pageviews data.
        country: Code of the country.
        df_dates: dataframe containing dates of first case, first death and lockdown.
        """

    # Get data
    df_covid = get_pageviews(df_pageviews, languages[country], "covid")
    df_politics = get_pageviews(df_pageviews, languages[country], POLITICS)

    # Select last available year of data
    df_covid = df_covid.loc[
        pd.date_range(pd.to_datetime("2019-08-01"), pd.to_datetime("2020-07-31"))
    ]
    df_politics = df_politics.loc[
        pd.date_range(pd.to_datetime("2019-08-01"), pd.to_datetime("2020-07-31"))
    ]

    # Plot subplots
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    # Rolling average across 14 days of COVID
    ax1.plot(df_covid.rolling(14).mean(), label="_nolabel_")

    # Rolling average across 14 days of politics
    ax2.plot(df_politics.rolling(14).mean(), label="_nolabel_")
    ax1.set_yscale("log")

    fig.suptitle(f"Daily Pageviews in {countries[country]}")
    ax1.set_title("COVID")
    ax2.set_title("Politics")
    ax1.set_ylabel("Pageviews")
    ax2.set_xlabel("Date")

    for ax in [ax1, ax2]:
        # plot the vertical line representing the date of the first case
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "1st case"].strftime("%Y-%m-%d")),
            color="black",
            linestyle="--",
            label="1st case",
        )
        # plot the vertical line representing the date of the first death
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "1st death"].strftime("%Y-%m-%d")),
            color="yellow",
            linestyle="--",
            label="1st death",
        )
        # plot the vertical line representing the date of the start of abnormal mobility
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "Mobility"].strftime("%Y-%m-%d")),
            color="red",
            linestyle="--",
            label="Abnormal mobility",
        )

        # plot the vertical line representing the date of return to normal mobility
        if country != "es":
            ax.axvline(
                x=pd.to_datetime(
                    df_dates.loc[country, "Normalcy"].strftime("%Y-%m-%d")
                ),
                color="green",
                linestyle="--",
                label="Normal mobility",
            )

    # Create single legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    num_lines = 3 if country == "es" else 4
    fig.legend(lines[:num_lines], labels[:num_lines])

    fig.show()


def plot_driving_walking_mobility(df, df_dates, country):
    """Plot mobility data for a given country. The data is split into driving and walking data and is plotted together.
    
    Args:
        df: dataframe containing apple mobility data.
        df_dates: dataframe containing dates of first case, first death and lockdown.
        country: code of the country
        """
    

    df_drive = df.loc[country, "driving"]
    df_walk = df.loc[country, "walking"]
    df_transport = pd.concat([df_drive, df_walk], axis=1)

    # Convert indices to datetime
    df_transport.index = pd.to_datetime(df_transport.index)

    # Keep only rows where date is between 2020-01-01 and < 2020-04-30
    df_transport = df_transport.loc[
        pd.to_datetime("2020-01-01") : pd.to_datetime("2020-04-30")
    ]

    # Create figure
    plt.figure(figsize=(7, 4), dpi=100)
    plt.plot(df_transport)

    plt.ylim(-100, 100)

    legends = ["Driving", "Walking", "First case", "First death"]

    dates = get_dates(df_dates, country)
    plot_dates(dates)

    if dates["Lockdown"] is not pd.NaT:
        legends.append("Lockdown")

    plt.title("Percentage of mobility change in {}".format(country))
    plt.xlabel("Date")
    plt.legend(legends)
    plt.xticks(rotation=45)
    plt.show()


def plot_google_mobility(df_mobility, df_dates, columns_dict, title, country):
    """Plot Google mobility data for a given country. The title arguement gives the column we want to plot.
    
    Args:
        df_transport: dataframe containing google mobility data 
        df_dates: dataframe containing dates of first case, first death and lockdown.
        columns_dict: dictionnary of titles mapped to column names of df_mobility
        title: title of the plot
        country: code of the country
    """
    # obtain the column name of the dataframe corresponding to the title
    column_name = columns_dict[title]

    # take the mobility data for the given country and the given title 
    df = df_mobility.loc[country][column_name]

    # plot the data
    plt.plot(df)
    plt.ylim(-100, 100)


    legends = ["Pageviews", "First case", "First death"]

    # retrieve the dates of the first case / death and lockdown
    dates = get_dates(df_dates, country)

    # plot those dates
    plot_dates(dates)

    # if no lockfown in the country, do not legend it
    if dates["Lockdown"] is not pd.NaT:
        legends.append("Lockdown")

    plt.title(f"{title} for {country}")
    plt.xlabel("Date")
    plt.ylabel(f"{column_name}")
    plt.legend(legends)
    plt.xticks(rotation=45)

    plt.show()



def plot_country_alignment(df, country, df_dates):
    """ Plot the political alignments of the country. The dataframes `df` is grouped by month, the mean and standard deviation of 
    each alignment are computed before plotting it.
    
    Args:
        df: dataframe containing the polling data
        country: code name of the country
        df_dates: dataframe containing dates of first case, first death and lockdown.
    """

    # Map each party to its political alignment
    df.columns = ["Sample Size"] + list(political_alignment[country].values())

    # Sum columns that belong to the same alignment
    df = df.groupby(df.columns, axis=1).sum()

    alignments = df.columns[1:]
    # Group the dataframe by month to obtain weighted mean and standard deviation of each alignment
    scores = group_date(df, alignments)

    # Perform linear interpolation if there are months with no data
    scores = linear_interpolation(scores, alignments)

    # Set the indexes
    idxs = scores.index.map(lambda x: x.strftime("%Y-%m"))

    # Plot data
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

    # plot the vertical line representing the date of the first death in the country
    plt.axvline(
        x=df_dates.loc[country, "1st death"].strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="1st death",
    )

    plt.legend(loc="lower left")
    plt.show()



def plot_mobility_pageviews_covid(df_transport1, df_pageviews, country, df_dates):
    """Plot mobility data for a given country. The data is split into driving and walking data and plotted together."""

    df_drive = df_transport1.loc[country, "driving"]
    df_walk = df_transport1.loc[country, "walking"]
    df_transport = pd.concat([df_drive, df_walk], axis=1)

    # Convert indices to datetime
    df_transport.index = pd.to_datetime(df_transport.index)

    # Keep only rows where date is between 2020-01-01 and 2020-04-30
    df_transport = df_transport.loc[
        pd.to_datetime("2020-01-01") : pd.to_datetime("2020-04-20")
    ]

    # Select last available year of data
    df_clipped = df_pageviews.loc[
        pd.date_range(pd.to_datetime("2020-01-01"), pd.to_datetime("2020-04-20"))
    ]

    legends = ["Driving", "Walking", "First case", "First death"]
    legends2 = ["First case", "First death"]
    ticks_x = [
        pd.to_datetime("2020-01-15"),
        pd.to_datetime("2020-02-01"),
        pd.to_datetime("2020-02-15"),
        pd.to_datetime("2020-03-01"),
        pd.to_datetime("2020-03-15"),
        pd.to_datetime("2020-04-01"),
        pd.to_datetime("2020-04-15"),
    ]

    plt.figure(1, figsize=(5, 7))
    plt.subplot(2, 1, 1)
    plt.plot(df_transport)
    plt.ylim(-100, 50)
    plt.ylabel("Percentage change [%]")
    plt.xlim(pd.to_datetime("2020-01-15"), pd.to_datetime("2020-04-20"))
    dates = get_dates(df_dates, country)
    plot_dates(dates)
    if dates["Lockdown"] is not pd.NaT:
        legends.append("Lockdown")
    plt.title("Percentage of mobility change in {}".format(country))
    plt.legend(legends)
    plt.xticks(ticks=ticks_x, labels=[], rotation=45)

    plt.subplot(2, 1, 2)
    # Rolling average across 14 days
    plt.plot(
        df_clipped.index,
        df_clipped.rolling(14).mean(),
        color="black",
        label="_nolegend_",
    )
    plt.title(f"Daily pageviews for covid related pages {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(ticks=ticks_x, rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()
    plt.xlim(pd.to_datetime("2020-01-15"), pd.to_datetime("2020-04-20"))
    dates = get_dates(df_dates, country)
    plot_dates(dates)
    plt.legend(legends2, loc="upper left")

    plt.show()


def plot_group_alignments(dfs, countries, df_dates, group_name):
    """Plot the political alignments of the group of countries. The dataframes in `dfs` are grouped by month,
    the mean and standard deviation of each alignment are computed  before concatenaiting the dataframes together, and plotting the result.
    
    Args:

        dfs: array of dataframes containing polling data
        countries: array of code of countries
        df_dates: dataframe containing dates of first case, first death and lockdown.
        group_name: name of the group """

    number_countries = len(countries)
    
    alignments = ['centre', 'left', 'right', 'centre-left', 'centre-right']


    for i in range(number_countries):

        # map each party to its political alignment
        dfs[i].columns = ["Sample Size"] + list(political_alignment[countries[i]].values())

        #sum columns that belong to the same alignment
        dfs[i] = dfs[i].groupby(dfs[i].columns, axis = 1).sum()


    for i in range(number_countries):

        # add columns filled with 0s of alignments that are not in the dataframe
        for alignment in alignments:
            if alignment not in dfs[i].columns:
                dfs[i][alignment] = 0

        # group the dataframe by month to obtain weighted mean and standard deviation of each alignment
        dfs[i] = group_date(dfs[i], alignments)

        # perform linear interpolation if there are some months with no data
        dfs[i] = linear_interpolation(dfs[i], alignments)


    # add the scores of each country in a single dataframe
    scores = dfs[0]
    for i in range(number_countries):
        scores = scores.add(dfs[i], fill_value=0)

    # divide by the number of countries
    scores = scores.div(number_countries)

    # set indexes
    idxs = scores.index.map(lambda x: x.strftime("%Y-%m"))

    # Plot the data
    for alignment in alignments:
        plt.fill_between(
            idxs,
            scores["avg_" + alignment] - scores["std_" + alignment],
            scores["avg_" + alignment] + scores["std_" + alignment],
            alpha=0.2,
        )
        plt.plot(idxs, scores["avg_" + alignment], label=alignment)

    plt.title(f"Political alignment popularity for {group_name}")
    plt.xlabel("Date")
    plt.xticks(idxs[::3], rotation=45)
    plt.ylabel("Percentage")


    # compute the latest first death date over countries
    latest_death = df_dates.loc[countries[0], "1st death"]
    for i in range(number_countries):
        if df_dates.loc[countries[i], "1st death"] > latest_death:
            latest_death = df_dates.loc[countries[i], "1st death"]

    # plot vertical line representing the latest first death
    plt.axvline(
        x=latest_death.strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="latest 1st death",
    )

    plt.legend(loc="lower left")
    plt.show()
