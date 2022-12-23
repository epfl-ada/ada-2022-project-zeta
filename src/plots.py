import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_processing import countries, languages
from src.helpers import *


POLITICS = "History and Society.Politics and government"


def plot_country_alignment(df, country, df_dates):
    """Plot the political alignments of the country. The dataframe `df` is grouped by month.
    The mean and standard deviation of each alignment are computed before plotting it.
    
    Args:
        df: dataframe containing the polling data.
        country: country code.
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

    # Set indices
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

    # Plot vertical line representing the date of the first death in the country
    plt.axvline(
        x=df_dates.loc[country, "1st death"].strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="1st death",
    )

    plt.legend(loc="lower left")
    plt.show()


def plot_dates(dates):
    """Plot dates of first case, first death and lockdown.

    Args: 
        dates: dataframe containing dates of first case, first death and lockdown.
    """

    # Retrieve dates
    first_case, first_death, lockdown = dates.values

    plt.axvline(x=first_case, color="green", linestyle="--")
    plt.axvline(x=first_death, color="red", linestyle="--")

    # Some countries do not have lockdown dates
    if lockdown is not pd.NaT:
        plt.axvline(x=lockdown, color="black", linestyle="--")


def plot_driving_walking_mobility(df, df_dates, country):
    """Plot mobility data for a given country. The data is split into driving and walking data and is plotted together.

    Args:
        df: dataframe containing Apple mobility data.
        df_dates: dataframe containing dates of first case, first death and lockdown.
        country: country code.
    """

    df_drive = df.loc[country, "driving"]
    df_walk = df.loc[country, "walking"]
    df_transport = pd.concat([df_drive, df_walk], axis=1)

    # Convert indices to datetime
    df_transport.index = pd.to_datetime(df_transport.index)

    # Keep only rows where date is between 2020-01-01 and 2020-04-30
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
        df_transport: dataframe containing Google mobility data .
        df_dates: dataframe containing dates of first case, first death and lockdown.
        columns_dict: dictionary of titles mapped to column names of df_mobility.
        title: plot title.
        country: country code.
    """

    # Obtain column name of the dataframe corresponding to the title
    column_name = columns_dict[title]

    # Take mobility data for the given country and given title
    df = df_mobility.loc[country][column_name]

    # Plot data
    plt.plot(df)
    plt.ylim(-100, 100)

    legends = ["Pageviews", "First case", "First death"]

    # Retrieve dates of the first case, first death and lockdown
    dates = get_dates(df_dates, country)

    # Plot dates
    plot_dates(dates)

    # If no lockdown in country, do not add to legend
    if dates["Lockdown"] is not pd.NaT:
        legends.append("Lockdown")

    plt.title(f"{title} for {country}")
    plt.xlabel("Date")
    plt.ylabel(f"{column_name}")
    plt.legend(legends)
    plt.xticks(rotation=45)

    plt.show()


def plot_group_alignments(dfs, countries, df_dates, group_name):
    """Plot the political alignments of the group of countries. The dataframes in `dfs` are grouped by month.
    The mean and standard deviation of each alignment are computed before concatenating the dataframes and plotting the result.
    
    Args:
        dfs: array of dataframes containing polling data.
        countries: list of country codes.
        df_dates: dataframe containing dates of first case, first death and lockdown.
        group_name: name of the group.
    """

    number_countries = len(countries)

    alignments = ["centre", "left", "right", "centre-left", "centre-right"]

    for i in range(number_countries):
        # Map each party to its political alignment
        dfs[i].columns = ["Sample Size"] + list(
            political_alignment[countries[i]].values()
        )

        # Sum columns that belong to the same alignment
        dfs[i] = dfs[i].groupby(dfs[i].columns, axis=1).sum()

    for i in range(number_countries):
        # Add columns filled with 0s for alignments that are not in the dataframe
        for alignment in alignments:
            if alignment not in dfs[i].columns:
                dfs[i][alignment] = 0

        # Group the dataframe by month to obtain weighted mean and standard deviation of each alignment
        dfs[i] = group_date(dfs[i], alignments)

        # Perform linear interpolation if there are some months with no data
        dfs[i] = linear_interpolation(dfs[i], alignments)

    # Add scores of each country in a single dataframe
    scores = dfs[0]
    for i in range(number_countries):
        scores = scores.add(dfs[i], fill_value=0)

    # Divide by number of countries
    scores = scores.div(number_countries)

    # Set indices
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

    plt.title(f"Political Alignment Popularity for {group_name}")
    plt.xlabel("Date")
    plt.xticks(idxs[::3], rotation=45)
    plt.ylabel("Percentage")

    # Compute the latest first death date among countries
    latest_death = df_dates.loc[countries[0], "1st death"]
    for i in range(number_countries):
        if df_dates.loc[countries[i], "1st death"] > latest_death:
            latest_death = df_dates.loc[countries[i], "1st death"]

    # Plot vertical line representing the latest first death
    plt.axvline(
        x=latest_death.strftime("%Y-%m"),
        color="black",
        linestyle="--",
        label="latest 1st death",
    )

    plt.legend(loc="lower left")
    plt.show()


def plot_mobility_pageviews_covid(df_transport, df_pageviews, country, df_dates):
    """Plot mobility data for a given country. The data is split into driving and walking data and plotted together.
    
    Args:
        df_transport: dataframe containing Apple mobility data
        df_pageviews: dataframe containing pageview data
        country: country code
        df_dates: `interventions.csv`
    """

    df_drive = df_transport.loc[country, "driving"]
    df_walk = df_transport.loc[country, "walking"]
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
    plt.title(f"Percentage Mobility Change in {countries[country]}")
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
    plt.title(f"Daily Pageviews for COVID Pages in {countries[country]}")
    plt.xlabel("Date")
    plt.xticks(ticks=ticks_x, rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()
    plt.xlim(pd.to_datetime("2020-01-15"), pd.to_datetime("2020-04-20"))
    dates = get_dates(df_dates, country)
    plot_dates(dates)
    plt.legend(legends2, loc="upper left")

    plt.show()


def plot_mobility_response(df, labels=None):
    """Plot response time and duration of reduced mobility period of countries. Optionally group the countries by `labels`.
    
    Args:
        df: dataframe containing `interventions.csv`
        labels: list of labels for countries
    """

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


def plot_pageviews(df, country, topic, df_dates):
    """Plot pageviews data for a given country and topic.

    Args:
        df: dataframe containing pageviews data.
        country: country code.
        topic: topic of the pageviews.
        df_dates: dataframe containing dates of first case, first death and lockdown."""

    # Select last available year of data
    df = df.loc[
        pd.date_range(pd.to_datetime("2019-08-01"), pd.to_datetime("2020-07-31"))
    ]

    # Rolling average across 14 days
    plt.plot(df.rolling(14).mean(), label="_nolegend_")

    if topic == POLITICS:
        plt.title(f"Daily Pageviews for Politics in {countries[country]}")

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Pageviews")
    plt.tight_layout()

    # Plot line representing starting date of abnormal mobility
    plt.axvline(
        x=pd.to_datetime(df_dates.loc[country, "Mobility"].strftime("%Y-%m-%d")),
        color="red",
        linestyle="--",
    )
    # Plot line representing return date of normal mobility
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
        country: country code.
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
    ax1.plot(df_covid.rolling(14).mean(), label="_nolabel_")
    ax2.plot(df_politics.rolling(14).mean(), label="_nolabel_")
    ax1.set_yscale("log")

    fig.suptitle(f"Daily Pageviews in {countries[country]}")
    ax1.set_title("COVID")
    ax2.set_title("Politics")
    ax1.set_ylabel("Pageviews")
    ax2.set_xlabel("Date")

    for ax in [ax1, ax2]:
        # Plot line representing date of first case
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "1st case"].strftime("%Y-%m-%d")),
            color="black",
            linestyle="--",
            label="1st case",
        )
        # Plot line representing date of first death
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "1st death"].strftime("%Y-%m-%d")),
            color="yellow",
            linestyle="--",
            label="1st death",
        )
        # Plot line representing starting date of abnormal mobility
        ax.axvline(
            x=pd.to_datetime(df_dates.loc[country, "Mobility"].strftime("%Y-%m-%d")),
            color="red",
            linestyle="--",
            label="Abnormal mobility",
        )
        # Plot line representing return date of normal mobility
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


def plot_polling_data(df, country, df_dates):
    """Plot polling data for a given country. The dataframe `df` is grouped by month, and the mean and standard deviation of each party is plotted.

    Args:
        df: dataframe containing polling data.
        country: country code.
        df_dates: dataframe containing dates of first case, first death and lockdown.
    """

    # Retrieve parties of the given country
    parties = df.columns[1:]

    # Group dataframe by month to obtain weighted mean and standard deviation of each alignment
    scores = group_date(df, parties)

    # Perform linear interpolation if there are some months with no data
    scores = linear_interpolation(scores, parties)

    # Set indices of the plot
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


def plot_silhouettes(X, k_min, k_max):
    """Plot Silhouette Scores for a range of clusters between `k_min` and `k_max` inclusive.

    Args:
        X: dataframe containing the data.
        k_min: minimum number of clusters to test.
        k_max: maximum number of clusters to test.
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
