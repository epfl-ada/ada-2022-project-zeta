# How Has Political Sentiment Changed Over the Course of the Pandemic?

EPFL Applied Data Analysis Course, Autumn 2022

Group project using the [Coronawiki](https://github.com/epfl-dlab/wiki_pageviews_covid) dataset.

View the data story [here](https://epfl-ada.github.io/ada-zeta-website/).

In order to rerun the notebook, Coronawiki dataset must be stored in `./data/`.

## Repository Overview

- `analysis.ipynb` - notebook containing our data analysis.

- `src/data_processing.py` - functions to process the data.

- `src/helpers.py` - various helper functions to get, query, and work with the data.

- `src/plots.py` - functions to produce plots.

## Abstract

In November 2019, the world heard the word CORONAVIRUS, which a few months later became COVID-19. This disease is caused by the SARS-CoV-2 virus which was first discovered in China. From that moment on, the spread of this virus continued to increase. Indeed, the pandemic spread from country to country, and is now present almost everywhere. Millions of people have been infected and over four hundred thousand have died!

The pandemic and its rapid spread caused fear worldwide. Sanitary measures and other restrictions were quickly put in place in many countries. Among the decisions taken by government leaders, those relating to individual movement were the most significant. During lockdowns, individuals became confined to their homes, and schools, nurseries, universities, canteens, restaurants, shops (except essential ones), etc., were closed.

At this point, we ask the following question: did the implementation of these restrictions impact political sentiment? How has political sentiment changed over the course of the pandemic?

In this context, it is natural to wonder whether an alignment between political decisions and people's behaviour has been achieved. Have people felt that their leaders need to be more in touch with their concerns and are unable to address the challenges at hand effectively? Has the COVID-19 pandemic, with its restrictions on daily life, sparked a renewed interest in politics among the population? As we examine this dynamic, it is also important to consider the role of the leading political party. Have they received support from the population, or have they faced disapproval for their actions or lack thereof?

## Research Questions

* Did the implementation of sanitary restrictions impact political sentiment?
* How has political sentiment changed over the course of the pandemic?
* Is there alignment between political decisions and people's behaviour during the COVID-19 pandemic?
* Has the COVID-19 pandemic increased people's interest in politics?
* How did people's opinions of political parties change in response to COVID-19 interventions?

## Additional Datasets

[European Opinion Polls as Open Data](https://filipvanlaenen.github.io/eopaod/)

This GitHub repository provides European opinion polls as open data. The data is available in CSV format. This will provide us a tool to observe how the popularity of political parties has evolved in different countries during the pandemic. We will also be able to observe any sudden increases or dips in popularity and see if they coincide with dates of any interventions.

The CSV data contains results from various national polls. We would like to work with a time series of the popularity of parties over time, so we will take the average over polls across a sliding window and interpolate a line graph from it.

## Methods

### Step 1: Data Scraping and Preprocessing

We create methods to pull the up-to-date polling data from the webpage and process both it and the Coronawiki data. With the data available in the format that we want, we are ready to perform an analysis of the data.

### Step 2: Characterise Government Responses

To evaluate how swift and stringent a government’s response to the pandemic was, we use the interventions data to determine how soon sanitary measures were put in place after COVID first appeared in the country, and for how long the population's mobility was affected. We use clustering methods to group countries based off of their response.

### Step 3: Explore polling data

With the scraped polling data, we plot how the popularity of the main political parties varied in the run-up to and during the pandemic. We observe how polling changed as a response to different "milestones" of the pandemic (such as the first case or the first death). We then use statistical testing to gather evidence to support our observations.

### Step 4: Analyse Wikipedia Pageview Data

We look at pageview data for COVID- and politics-related pages, overlayed with key dates from the interventions data. Using visual methods, we look for similar behaviours that occur between populations in countries within the same group and observe the exponential increase in interest COVID experienced in the first stages of the pandemic.

### Step 5: Explore Mobility Data

Overlayed with the same key dates, we plot how the mobility changed over the course of the pandemic. Simultaneously, we can compare the progression of the mobility change with the time series of the Wikipedia pageviews of COVID-related pages to see if the two are related.

## Organisation

Anton: Problem formulation, performing statistical analysis, coding data analysis methods

Olena: Scraping the data, preparing the final data story

Tiavina: Plotting graphs during data analysis, writing up the data story

Thomas: Performing data analysis, coding data processing methods
