from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


# Political parties per country
parties = {
    "de": {
        "FDP": "FDP",
        "SPD": "SPD",
        "Union": "CDU/CSU *",
        "Alternative für Deutschland": "AfD",
        "BÜNDNIS 90/DIE GRÜNEN": "GRÜNE",
    },
    "dk": {
        "Socialdemokraterne": "SD *",
        "Venstre": "Venstre",
        "Liberal Alliance": "LA",
        "Socialistisk Folkeparti": "SF",
        "Det Konservative Folkeparti": "Konservative",
    },
    "es": {
        "Partido Popular": "PP",
        "Partido Socialista Obrero Español": "PSOE *",
        "Unidos Podemos": "UP",
        "Ciudadanos–Partido de la Ciudadanía": "Cs",
        "Vox": "Vox",
    },
    "fi": {
        "Suomen Keskusta": "Kesk",
        "Kansallinen Kokoomus": "Kok",
        "Perussuomalaiset": "PS",
        "Suomen Sosialidemokraattinen Puolue": "SDP *",
        "Vihreä liitto": "Vihr",
    },
    "fr": {
        "La France insoumise": "LFI",
        "Parti socialiste": "PS",
        "La République en marche–Mouvement démocrate": "LREM *",
        "Les Républicains": "LR",
        "Rassemblement national": "RN",
    },
    "it": {
        "Partito Democratico": "PD (S&D)",
        "Movimento 5 Stelle": "M5S (NI) *",
        "Forza Italia": "FI (EPP)",
        "Lega Nord": "LEGA (ID)",
        "Fratelli d’Italia": "FdI (ECR)",
    },
    "nl": {
        "Volkspartij voor Vrijheid en Democratie": "VVD *",
        "Partij voor de Vrijheid": "PVV",
        "Christen-Democratisch Appèl": "CDA",
        "Democraten 66": "D66",
        "Partij van de Arbeid": "PvdA",
    },
    "no": {
        "Sosialistisk Venstreparti": "SV",
        "Arbeiderpartiet": "AP",
        "Senterpartiet": "SP",
        "Høyre": "Høyre *",
        "Fremskrittspartiet": "FrP",
    },
    "se": {
        "Sveriges socialdemokratiska arbetareparti": "SSAP *",
        "Moderata samlingspartiet": "MSP",
        "Sverigedemokraterna": "SD",
        "Centerpartiet": "CP",
        "Vänsterpartiet": "VP",
    },
}

# Main ruling party at the start of the pandemic per country
main_party = {
    "de": "CDU/CSU *",
    "dk": "SD *",
    "es": "PSOE *",
    "fi": "SDP *",
    "fr": "LREM *",
    "it": "M5S (NI) *",
    "nl": "VVD *",
    "no": "Høyre *",
    "se": "SSAP *",
}

# Political alignment of each party
political_alignment = {
    "de": {
        "FDP": "centre-right",
        "SPD": "centre-left",
        "CDU/CSU *": "centre-right",
        "AfD": "right",
        "GRÜNE": "centre-left",
    },
    "dk": {
        "SD *": "centre-left",
        "Venstre": "centre-right",
        "LA": "centre-right",
        "SF": "left",
        "Konservative": "centre-right",
    },
    "es": {
        "PP": "centre-right",
        "PSOE *": "centre-left",
        "UP": "left",
        "Cs": "centre",
        "Vox": "right",
    },
    "fi": {
        "Kesk": "centre",
        "Kok": "centre-right",
        "PS": "right",
        "SDP *": "centre-left",
        "Vihr": "centre-left",
    },
    "fr": {
        "LFI": "left",
        "PS": "centre-left",
        "LREM *": "centre",
        "LR": "centre-right",
        "RN": "right",
    },
    "it": {
        "PD (S&D)": "centre-left",
        "M5S (NI) *": "right",
        "FI (EPP)": "centre-right",
        "LEGA (ID)": "right",
        "FdI (ECR)": "right",
    },
    "nl": {
        "VVD *": "centre-right",
        "PVV": "right",
        "CDA": "centre-right",
        "D66": "centre",
        "PvdA": "centre-left",
    },
    "no": {
        "SV": "left",
        "AP": "centre-left",
        "SP": "centre-left",
        "Høyre *": "centre-right",
        "FrP": "right",
    },
    "se": {
        "SSAP *": "centre-left",
        "MSP": "centre-right",
        "SD": "right",
        "CP": "centre",
        "VP": "left",
    },
}


def calculate_mean(col, coeff):
    """Calculate the weighted mean of a list.

    Args:
        col: list of values
        coeff: list of coefficient

    Returns: weighted mean
    """

    return (col * coeff).sum() / coeff.sum()


def calculate_std(col, coeff):
    """Calculate the standard deviation of a list from a weighted mean.
    
    Args:
        col: list of values
        coeff: list of coefficient

    Returns: weighted standard deviation
    """

    return (
        (coeff * (col - calculate_mean(col, coeff)) ** 2).sum() / coeff.sum()
    ) ** 0.5


def flatten(list):
    """Flatten a list of lists.
    
    Args:
        list: list of lists.
    
    Returns: flattened list."""

    return [item for sublist in list for item in sublist]


def get_dates(df_dates, country):
    """Get first case, first death and lockdown dates for a given country.
    
    Args:
        df_dates: dataframe with dates.
        country: country code.
        
    Returns: Series of dates.
    """

    return pd.to_datetime(df_dates.loc[country][["1st case", "1st death", "Lockdown"]])


def get_pageviews(df, lang, topic, measure="sum"):
    """Combines the views from desktop and mobile Wikipedia pages for a given language and topic.

    Args:
        df: pageviews (dataframe).
        lang: language code (string).
        topic: page topic or "covid" (string).
        measure: "sum" or "percent".

    Returns:
        Series of dates and topic pageviews.
    """

    # Select desktop and mobile pageviews
    if topic == "covid":
        df1 = df[lang]["covid"][measure]
        df2 = df[lang + ".m"]["covid"][measure]
    else:
        df1 = df[lang]["topics"][topic][measure]
        df2 = df[lang + ".m"]["topics"][topic][measure]

    # defaultdict in case pageview count for a specific day only appears in one series
    total_views = defaultdict(int, df1)

    # Combine pageviews
    if measure == "sum":
        for date, views in df2.items():
            total_views[date] += views

    elif measure == "percent":
        if topic == "covid":
            df1_counts = defaultdict(float, df[lang]["covid"]["sum"])
            df2_counts = defaultdict(float, df[lang + ".m"]["covid"]["sum"])
        else:
            df1_counts = defaultdict(float, df[lang]["topics"][topic]["sum"])
            df2_counts = defaultdict(float, df[lang + ".m"]["topics"][topic]["sum"])

        for date, views in df2.items():
            total_views[date] = (
                (total_views[date] * df1_counts[date] + views * df2_counts[date])
                / (df1_counts[date] + df2_counts[date])
                if df1_counts[date] + df2_counts[date] != 0
                else 0
            )

    # Convert to dataframe
    total_views = pd.DataFrame(total_views.items(), columns=["date", "pageviews"])

    # Extract date
    total_views["date"] = pd.to_datetime(total_views["date"]).dt.date

    # Set index
    total_views = total_views.set_index("date")

    return total_views


def get_polling_data(country, monthly=True):
    """Get polling data for a given country.

    Args:
        country: country code.
    
    Returns:
        Polling data.
    
    Raises:
        KeyError: data unavailable for given country.
    """

    # Country codes where polling data is available
    available_data = ["de", "dk", "es", "fi", "fr", "it", "nl", "no", "se"]

    # Error if data unavailable
    if country not in available_data:
        raise KeyError("Data unavailable for given country.")

    # Data location on web
    url = f"https://filipvanlaenen.github.io/eopaod/{country}.csv"

    # Read data
    df = pd.read_csv(url, on_bad_lines="skip")

    # Assign poll date at end of fieldwork
    df["Date"] = pd.to_datetime(df["Fieldwork End"])
    # Ignore day of month
    if monthly:
        df["Date"] = df["Date"].dt.to_period("M")

    # Keep polls between 2019 and April 2021
    df = df[("2019-01" <= df["Date"]) & (df["Date"] <= "2021-04")]

    # Keep relevant columns
    df = df[["Date", "Sample Size"] + list(parties[country].keys())]
    # Rename columns
    df.columns = ["Date", "Sample Size"] + list(parties[country].values())

    # Cast to int
    df["Sample Size"] = pd.to_numeric(df["Sample Size"], errors="coerce")

    # Convert percentages into values between 0 and 1
    for col in df.iloc[:, 2:]:
        df[col] = (
            pd.to_numeric(df[col].replace("%", "", regex=True), errors="coerce") / 100
        )

    # Set and sort index
    df = df.set_index("Date").sort_index()

    return df


def group_date(df, columns):
    """Group the dataframe df by month, and compute the weighted mean and standard deviation for each column in columns.
    
    Args:
        df: dataframe.
        columns: list of columns of interest to compute the weighted mean and standard deviation.
       
    Returns: dataframe containing the weighted mean and standard deviation for each date.
    """

    # Group dataframe by date
    df_grouped = df.groupby("Date")

    # Create dataframe containing the average of each columns over a month
    avg = df_grouped.apply(
        lambda x: pd.Series(
            {
                "avg_" + column: calculate_mean(x[column], x["Sample Size"])
                for column in columns
            }
        )
    )

    # Create dataframe containing the standard deviation of each columns over a month
    std = df_grouped.apply(
        lambda x: pd.Series(
            {
                "std_" + column: calculate_std(x[column], x["Sample Size"])
                for column in columns
            }
        )
    )

    # Concatenate dataframes with respect to the date
    return pd.concat([avg, std], axis=1)


def linear_interpolation(df, columns):
    """Fill the missing values in the dataframe by linear interpolation.
    
    Args:   
        df: dataframe.
        columns: list of columns of interest.
    
    Returns: dataframe with missing values filled by linear interpolation.
    """

    # List of months
    date_list = [
        "2019-01",
        "2019-02",
        "2019-03",
        "2019-04",
        "2019-05",
        "2019-06",
        "2019-07",
        "2019-08",
        "2019-09",
        "2019-10",
        "2019-11",
        "2019-12",
        "2020-01",
        "2020-02",
        "2020-03",
        "2020-04",
        "2020-05",
        "2020-06",
        "2020-07",
        "2020-08",
        "2020-09",
        "2020-10",
        "2020-11",
        "2020-12",
        "2021-01",
        "2021-02",
        "2021-03",
        "2021-04",
    ]

    for i, date in enumerate(date_list):

        # For each date in date_list, check whether it is in dataframe
        if date not in df.index and i != 0:

            # Save last date that is in dataframe
            previous_date = date_list[i - 1]

            # Search for next date that is in dataframe
            new_index = i
            while (
                date_list[new_index] not in df.index and new_index < len(date_list) - 1
            ):
                new_index += 1

            next_date = date_list[new_index]

            # Number of invalid dates between valid ones
            number_dates = new_index - i

            # Perform a linear interpolation between date and new_date
            intervals = np.linspace(0, 1, number_dates + 2)
            nb_interval = 1
            while i != new_index:
                curr_date = date_list[i]
                curr_t = intervals[nb_interval]

                for column in columns:
                    df.loc[curr_date, "avg_" + column] = (1 - curr_t) * df.loc[
                        previous_date, "avg_" + column
                    ] + curr_t * df.loc[next_date, "avg_" + column]
                    df.loc[curr_date, "std_" + column] = (1 - curr_t) * df.loc[
                        previous_date, "std_" + column
                    ] + curr_t * df.loc[next_date, "std_" + column]

                nb_interval += 1
                i += 1

    # Sort resulting dataframe
    return df.sort_index()


def polling_data_ttest(country, df_dates):
    """Perform a t-test determining whether or not there is a statistically significant
    difference in polling immediately before and after the start of the pandemic.
    
    Args:
        country: country code.
        df_dates: `interventions.csv`.
    
    Returns: p-value from independent t-test sampling.
    """

    # Get polls for country
    df = get_polling_data(country, monthly=False)[main_party[country]]

    # Get date of first death
    cutoff_date = pd.to_datetime(
        df_dates.loc[country, "1st death"].strftime("%Y-%m-%d")
    )

    # Polls 2 months before pandemic
    pre_covid_polls = df[
        (cutoff_date - pd.Timedelta(weeks=8) < df.index) & (df.index <= cutoff_date)
    ]
    # Polls 2 months into pandemic
    post_covid_polls = df[
        (cutoff_date < df.index) & (df.index <= cutoff_date + pd.Timedelta(weeks=8))
    ]

    # Calculate t-test
    return stats.ttest_ind(pre_covid_polls, post_covid_polls).pvalue
