"""Functions to scrape data."""

import pandas as pd


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
