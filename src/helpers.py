"""Various helper functions to get and query data."""

from collections import defaultdict

import pandas as pd


# Political parties per country
parties = {
    "de": {
        "FDP": "FDP",
        "SPD": "SPD",
        "Union": "CDU/CSU *",
        "Alternative für Deutschland": "AfD",
        # "DIE LINKE": "DIE LINKE",
        "BÜNDNIS 90/DIE GRÜNEN": "GRÜNE",
    },
    "dk": {
        "Socialdemokraterne": "SD *",
        # "Dansk Folkeparti": "DF",
        "Venstre": "Venstre",
        # "Enhedslisten–De Rød-Grønne": "EDRG",
        "Liberal Alliance": "LA",
        # "Alternativet": "Alternativet",
        # "Radikale Venstre": "RV",
        "Socialistisk Folkeparti": "SF",
        "Det Konservative Folkeparti": "Konservative",
        # "Kristendemokraterne": "Kristendemokraterne",
        # "Nye Borgerlige": "NB",
        # "Borgerlisten": "Borgerlisten",
        # "Stram Kurs": "SK",
        # "Veganerlisten": "Veganerlisten",
        # "Moderaterne": "Moderaterne",
        # "Frie Grønne": "FG",
        # "Danmarksdemokraterne": "DD",
    },
    "es": {
        "Partido Popular": "PP",
        "Partido Socialista Obrero Español": "PSOE *",
        "Unidos Podemos": "UP",
        "Ciudadanos–Partido de la Ciudadanía": "Cs",
        # "Esquerra Republicana de Catalunya–Catalunya Sí": "ERC",
        "Vox": "Vox",
    },
    "fi": {
        "Suomen Keskusta": "Kesk",
        "Kansallinen Kokoomus": "Kok",
        "Perussuomalaiset": "PS",
        "Suomen Sosialidemokraattinen Puolue": "SDP *",
        "Vihreä liitto": "Vihr",
        # "Vasemmistoliitto": "Vas",
        # "Svenska folkpartiet i Finland": "SFP",
        # "Kristillisdemokraatit": "KD",
        # "Sininen tulevaisuus": "Sin",
        # "Liike Nyt": "Liik",
    },
    "fr": {
        # "Lutte Ouvrière": "LO",
        # "Nouveau Parti anticapitaliste": "NPA",
        # "Parti communiste français": "PCF",
        "La France insoumise": "LFI",
        # "Génération·s, le mouvement": "GÉNÉRATION·S",
        # "Europe Écologie Les Verts": "EÉLV",
        "Parti socialiste": "PS",
        "La République en marche–Mouvement démocrate": "LREM *",
        # "Agir, la droite constructive–Union des démocrates et indépendants": "AGIR",
        # "Résistons!": "RÉSISTONS!",
        "Les Républicains": "LR",
        # "Debout la France": "DLF",
        "Rassemblement national": "RN",
        # "Les Patriotes": "PATRIOTES",
        # "Union populaire républicaine": "UPR",
        # "Mouvement des gilets jaunes": "GILETS JAUNES",
        # "L’Engagement": "ENGAGEMENT",
        # "Reconquête": "RECONQUÊTE",
        # "Walwari": "WALWARI",
    },
    "it": {
        "Partito Democratico": "PD (S&D)",
        # "Più Europa": "+E (RE)",
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
        # "GroenLinks": "GL",
        # "Socialistische Partij": "SP",
        "Partij van de Arbeid": "PvdA",
        # "ChristenUnie": "CU",
        # "Partij voor de Dieren": "PvdD",
        # "Staatkundig Gereformeerde Partij": "SGP",
        # "Forum voor Democratie": "FvD",
        # "Juiste Antwoord 2021": "JA21",
        # "Volt Europa": "Volt",
        # "BoerBurgerBeweging": "BBB",
    },
    "no": {
        # "Rødt": "Rødt",
        "Sosialistisk Venstreparti": "SV",
        # "Miljøpartiet De Grønne": "MDG",
        "Arbeiderpartiet": "AP",
        "Senterpartiet": "SP",
        # "Venstre": "Venstre",
        # "Kristelig Folkeparti": "KRF",
        "Høyre": "Høyre *",
        "Fremskrittspartiet": "FrP",
    },
    "se": {
        "Sveriges socialdemokratiska arbetareparti": "SSAP *",
        "Moderata samlingspartiet": "MSP",
        "Sverigedemokraterna": "SD",
        "Centerpartiet": "CP",
        "Vänsterpartiet": "VP",
        # "Kristdemokraterna": "KD",
        # "Liberalerna": "Liberalerna",
        # "Miljöpartiet de gröna": "MP",
    },
}


def flatten(list):
    """Flatten a list of lists."""

    return [item for sublist in list for item in sublist]


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
    df["Date"] = pd.to_datetime(df["Fieldwork End"]).dt.to_period("M")

    # Keep polls starting from 2019
    df = df[df["Date"] >= "2019-01"]

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
