"""Various helper functions to get and query data."""

from collections import defaultdict

import pandas as pd


def get_pageviews(df, lang, topic, measure="sum"):
    """Combines the views from desktop and mobile Wikipedia pages for a given language and topic.

    Args:
        df: pageviews (dataframe).
        lang: language code (string).
        topic: page topic or "covid" (string).
        measure: "sum" or "percent"

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


def get_polling_data(country):
    """Get polling data for a given country.

    Args:
        country: country code
    
    Returns:
        Polling data.
    
    Raises:
        KeyError: data unavailable for given country
    """

    available_data = ["de", "dk", "es", "fi", "fr", "it", "nl", "no", "se"]

    if country not in available_data:
        raise KeyError("Data unavailable for given country.")

    url = f"https://filipvanlaenen.github.io/eopaod/{country}.csv"

    df = pd.read_csv(url, on_bad_lines="skip")

    return df
