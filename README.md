# How Has Political Sentiment Changed Over the Course of the Pandemic?

## Abstract

During the COVID-19 pandemic, we saw that governments implemented different sanitary measures in order to reduce the strain on their health services. Governments balanced imposing sufficiently stringent measures while at the same time not depriving the populace of too many liberties in order to not lose the public’s support.

The aim of this project is to analyse how public support for governments and interest in politics changed depending on the response of the government to the pandemic. We would also like to find out which approaches in dealing with the pandemic inspired confidence in the public. We aim to classify government responses based on their swiftness and severity, and then explore how the population's mobility patterns changed. We will then be able to gauge interest in politics through Wikipedia pageview data and evaluate support in the government through polling and election data.

## Research Questions

* Do lax or stringent sanitary measures impact public support in the government? Does this occur everywhere or only in certain regions?
* Is there a strategy that governments could have taken that retains public support?
* Are people more likely to flout lockdown rules if they are dissatisfied with the government?
* Did interest in politics alter during the pandemic and, if so, has this change persisted?

## Proposed Additional Datasets

[European Opinion Polls as Open Data](https://filipvanlaenen.github.io/eopaod/)

This GitHub repository provides European opinion polls as open data. The data is available in CSV format. This will provide us a tool to observe how the popularity of political parties has evolved in different countries during the pandemic. We will also be able to observe any sudden increases or dips in popularity and see if they coincide with dates of any interventions.

The CSV data contains results from various national polls. We would like to work with a time series of the popularity of parties over time, so we will take the average over polls across a sliding window and interpolate a line graph from it.

## Methods

### Step 1: Data Scraping and Preprocessing

We create methods to pull the up-to-date polling data from the webpage and process both it and the Coronawiki data. With the data available in the format that we want, we are ready to perform an analysis of the data.

### Step 2: Explore Mobility Data

TODO

### Step 3: Characterise Government Responses

To evaluate how swift and stringent a government’s response to the pandemic was, we use the interventions data to determine how soon sanitary measures were put in place after COVID first appeared in the country, and for how long the population's mobility was affected. We use clustering methods to group countries based off of their response.

### Step 4: Explore polling data

With the scraped polling data, we plot how the popularity of the main political parties varied in the run-up to and during the pandemic. We observe how polling changed as a response to different "milestones" of the pandemic (such as the first case or the first death). We then use statistical testing to gather evidence to support our observations.

### Step 5: Analyse Wikipedia Pageview Data

We look at pageview data for COVID- and politics-related pages, overlayed with key dates from the interventions data. Using visual methods, we look for similar behaviours that occur between populations in countries within the same group and observe the exponential increase in interest COVID experienced in the first stages of the pandemic.

## Organisation

Anton: Problem formulation, performing statistical analysis, coding data analysis methods

Olena: Scraping the data, preparing the final data story

Tiavina: Plotting graphs during data analysis, writing up the data story

Thomas: Performing data analysis, coding data processing methods
