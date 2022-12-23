import pandas as pd


# Mapping country codes to country names
countries = {
    "de": "Germany",
    "dk": "Denmark",
    "es": "Spain",
    "fi": "Finland",
    "fr": "France",
    "it": "Italy",
    "jp": "Japan",
    "kr": "South Korea",
    "nl": "Netherlands",
    "no": "Norway",
    "rs": "Serbia",
    "se": "Sweden",
}

# Mapping country codes to language codes of corresponding Wikipedia language
languages = {
    "de": "de",
    "dk": "da",
    "es": "ca",
    "fi": "fi",
    "fr": "fr",
    "it": "it",
    "jp": "ja",
    "kr": "ko",
    "nl": "nl",
    "no": "no",
    "rs": "sr",
    "se": "sv",
}


def process_interventions_data(df):
    """Process data loaded from `interventions.csv`.
    
    Args:
        df: `interventions.csv` dataframe.
    
    Returns:
        Augmented interventions data.
    """

    # Replace language codes with country codes
    df = (
        df.rename_axis("country")
        .rename(
            index={
                "da": "dk",
                "sr": "rs",
                "sv": "se",
                "ko": "kr",
                "ca": "es",
                "ja": "jp",
            }
        )
        .sort_index()
    )

    # Convert columns to datetime
    df = df.apply(pd.to_datetime)

    # Period between date of 1st case and date of first imposed measure
    df["Response time"] = (
        df[["School closure", "Public events banned", "Lockdown"]].min(axis=1)
        - df["1st case"]
    ).dt.days

    # Duration of abnormal mobility
    df["Reduced mobility"] = (df["Normalcy"] - df["Mobility"]).dt.days

    return df


def process_mobility_data(df, countries=countries):
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


def process_transport_data(df, countries=countries):
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
