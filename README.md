# How has political sentiment changed over the course of the pandemic?

## Contents

- `additional_datasets.ipynb` - notebook showcasing additional datasets and demonstrating that we can use the data in our analysis

- `coronawiki.ipynb` - notebook containing an exploration and transformation of the CoronaWiki datasets

- `data_processing.py` - helper functions to transform and query the CoronaWiki datasets

- `initial_analysis.ipynb` - notebook containing a potential initial analysis of our data, showing that we are able to work with it

- `japan_csv_data.html` - HTML script to download polling data for Japan

## Abstract

During the COVID-19 pandemic, we saw that governments implemented different sanitary measures in order to reduce the strain on their health services. Governments balanced imposing sufficiently stringent measures while at the same time not depriving the populace of too many liberties in order to not lose the public’s support.

The aim of this project is to analyse how public support for governments and interest in politics changed depending on the response of the government to the pandemic. We would also like to find out which approaches in dealing with the pandemic inspired confidence in the public. We aim to classify government responses based on their swiftness and severity, and then explore how the population's mobility patterns changed. We will then be able to gauge interest in politics through Wikipedia pageview data and evaluate support in the government through polling and election data.

## Research Questions

* Do lax or stringent sanitary measures impact public support in the government? Does this occur everywhere or only in certain regions?
* Is there a strategy that governments could have taken that retains public support?
* Are people more likely to flout lockdown rules if they are dissatisfied with the government?
* Did interest in politics alter during the pandemic and, if so, has this change persisted?

## Proposed Additional Datasets

[Europe Elects](https://europeelects.eu/)

The Europe Elects website gives us access to databases of polling data on the popularity of political parties in various European countries since January 2018. The data is available to download in CSV format. This will give us an additional tool to observe how the popularity of political parties has evolved during the pandemic in the different countries. We will also be able to observe any sudden increases or dips in popularity and see if they coincide with dates of any interventions.

The CSV data contains results from various national polls. We would like to work with a time series of the popularity of parties over time, so we will take the average over polls and interpolate a line graph from it.

Polling data for Japan and South Korea are not available from this database.

---

[PolitPro](https://politpro.eu/en/japan)

PolitPro provides polling data in different countries since January 2018. We cannot as easily extract CSV data as from Europe Elects, but we are able to get polling data for Japan.

The website does not allow us to download the data in CSV format. The page's HTML source contains a variable in which data is stored as well as a diagram for the data. Using developer tools and JavaScript, we can get the variable that contains the data for the popularity of political parties. Running a HTML script that we wrote, the data is parsed and a CSV file containing the data is produced which we can then use. Ultimately, this approach is rather convoluted and the data that we have scraped is not as detailed as the data from Europe Elects, but we carried it out to show that it would be possible to get polling data for Japan if needed in our project.

## Methods

### Step 1: Data scraping and preprocessing

With the downloaded polling data, we will ensure that our data is ready to be used in our analysis. As part of this milestone, we have already transformed the CoronaWiki data to be easily queried, and we have managed to extract time series from the polling data. As such, this step should not require much additional work.

### Step 2: Characterise government responses

To evaluate how swift and stringent a government’s response to the pandemic was, we can use the interventions data to determine how soon sanitary measures were put in place after COVID first appeared in the country. We can also use the same data to see what measures were put in place, since not all countries imposed the same measures (i.e. not all countries had full lockdowns), and for how long mobility was affected. We could potentially use clustering methods for this, and this will allow us to group countries based on their COVID strategies.

### Step 3: Explore interest in politics

We can then look at the Wikipedia data for each country. We can look at the pageview time series for topics relating to government and politics, and the country’s region. An increase in both would suggest that people in the country are looking more at pages relating to their politicians and government. Since we have pageview counts from 2018, we can also see if interest in the long-term in this topic has changed using t-tests on the page views before and after the pandemic.

### Step 4: Explore polling data

We can also look at the polling data to see if government support changes following the implementation of different measures. As a hypothesis, we might observe that government support dips a week into lockdown, and this could be detected through visualisations of the polling data with intervention dates overlayed. We would detect these changes through visual methods and could confirm them using t-tests.

### Step 5: Explore mobility data

We can also plot the mobility data with the intervention dates overlayed. This will allow us to see how populations reacted to the pandemic (i.e. did mobility decrease only once governments imposed mobility restrictions, or did people preemptively reduce their own mobility). Alongside the polling data, this could potentially allow us to deduce whether populations had trust in their government. For instance, if government support was high and mobility was reduced before restrictions were imposed (i.e. when governments issued guidance), then it can suggest a high level of trust in the government.

## Proposed Timeline

Week 1 - Steps 1 & 2

Week 2 - Step 3

Week 3 - Step 4

Week 4 - Step 5

Week 5 - Wrapping up and creating website

## Organisation

Anton - Statistical analysis

Olena - Data scraping and analysis

Tiavina - Visualisations and analysis

Thomas - Data scraping and analysis
