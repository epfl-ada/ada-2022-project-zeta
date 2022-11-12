"""Functions to process and query data."""

from collections import defaultdict

import pandas as pd


def process_interventions_data(df):
    """Process data loaded from `interventions.csv`.
    
    Args:
        df: `interventions.csv` dataframe.
    
    Returns:
        Augmented interventions data.
    """

    # Convert columns to datetime
    df = df.apply(pd.to_datetime)

    # Period between date of 1st case and date of first imposed measure
    df["Response time"] = (
        df[["School closure", "Public events banned", "Lockdown"]].min(axis=1)
        - df["1st case"]
    )

    # Duration of abnormal mobility
    df["Reduced mobility"] = df["Normalcy"] - df["Mobility"]

    return df


def process_mobility_data(df, countries):
    """Process data loaded from `Global_Mobility_Report.csv.gz`.

    Args:
        df: `Global_Mobility_Report.csv.gz` dataframe.
        countries: dictionary of country code and country name pairs.
    
    Returns:
        Processed mobility data.
    """

    # Lower case country code
    df["country_region_code"] = df["country_region_code"].str.lower()

    # Select only entries in countries of interest
    df = df[df["country_region_code"].isin(countries.keys())].reset_index(drop=True)

    # Remove subregion entries
    df = df.loc[
        (df.sub_region_1.isnull())
        & (df.sub_region_2.isnull())
        & (df.metro_area.isnull())
    ]

    # Drop unnecessary columns
    df = df.drop(
        [
            "country_region",
            "sub_region_1",
            "sub_region_2",
            "metro_area",
            "iso_3166_2_code",
            "census_fips_code",
        ],
        axis=1,
    )

    # Rename columns
    df = df.rename(columns={"country_region_code": "country"})

    # Set country code and date as index
    df = df.set_index(["country", "date"]).sort_index()

    return df


def process_transport_data(df, countries):
    """Process data loaded from `applemobilitytrends-2020-04-20.csv.gz`.

    Args:
        df: `applemobilitytrends-2020-04-20.csv.gz` dataframe.
        countries: dictionary of country code and country name pairs.
    
    Returns:
        Processed transport data.
    """

    # Select only rows in countries of interest
    df = df.loc[df.region.isin(countries.values())].reset_index(drop=True)

    # Rename columns
    df = df.rename(columns={"region": "country"})

    # Convert country name to country code
    df["country"] = df["country"].apply(
        lambda x: list(countries.keys())[list(countries.values()).index(x)]
    )

    # Convert from percentage to percentage change
    for col in df.columns[3:]:
        df[col] -= 100

    # Convert dataframe from wide to long format
    df = df.melt(
        id_vars=["country", "transportation_type"],
        value_vars=df.columns[3:],
        var_name="date",
        value_name="percent_change_from_baseline",
    )

    # Set index columns
    df = df.set_index(["country", "transportation_type", "date"]).sort_index()

    return df


def get_pageviews(df, lang, topic):
    """Combines the views from desktop and mobile Wikipedia pages for a given language and topic.

    Args:
        df: pageviews (dataframe).
        lang: language code (string).
        topic: page topic (string).

    Returns:
        Series of dates and topic pageviews.
    """

    # Select desktop and mobile pageviews
    df1 = df[lang]["topics"][topic]["sum"]
    df2 = df[lang + ".m"]["topics"][topic]["sum"]

    # defaultdict in case pageview count for a specific day only appears in one series
    total_views = defaultdict(int, df1)

    # Combine pageviews
    for date, views in df2.items():
        total_views[date] += views

    # Convert to dataframe
    total_views = pd.DataFrame(total_views.items(), columns=["date", "pageviews"])

    # Extract date
    total_views["date"] = pd.to_datetime(total_views["date"]).dt.date

    # Set index
    total_views = total_views.set_index("date")

    return total_views
