# 10**6 - Final Degree Thesis

This repository contains the code for my final degree thesis. The goal is to build a tool that empowers investors to obtain powerful insights at a glance. By leveraging multimodal, data-driven analysis, the system aims to support smarter investment decisions and potentially achieve better returns compared to traditional techniques.

Read my final degree thesis [here]().
----

## Project description 

### First Deliverable: Newsletter Automation

A daily newsletter is sent (automatically at 9.00 a.m UTC) containing the top-10 most relevant stock-related news articles of today.

The pipeline is the following. First fetch Expansión RSS (Mercados, Ahorro, Empresas). Then preprocess and filter relevant news. Convert to html, embed an image and finally send the newsletter. The workflow can be executed both locally and from GitHub Actions.

#### Newsletter example
![image](imgs/p_newsletter.png)

### Second Deliverable: Database Ingestion and Time Series Modeling

From Monday to Friday after the Spanish market closes, Open, High, Low, Close, and Volume (OHLCV) data are automatically fetched for all IBEX35 companies and appended to the database.

We then apply ARIMA models to BBVA.MC prices and log returns over short-term and long-term horizons. Prices are non-stationary, while log returns behave as white noise, making ARIMA a suitable baseline model. Forecast variance increases over time, making long-term forecasts unreliable. 

#### Arima forecast 
![image](imgs/arima.png)

### Third Deliverable: Further Time Series Modeling and news Ingestion

We then incorporate a GARCH model to better capture volatility dynamics. Variance is now not assumed constant and prediction residuals are heteroskedastic, reflecting the clustering of shocks over time. Confidence bands are provided around the forecasts, illustrating how uncertainty expands as the prediction horizon increases. 

#### Garch forecast
![image](imgs/garch.png)

Stock-related news articles are processed on a daily basis. A Large Language Model (LLM) extracts mentioned companies, performs sentiment analysis and determines relevance. Then they are stored. 

**How is information stored?**
Initially, a GitHub automation bot fetched data, appended it to a local database, and committed updates directly to the repository. The system has now evolved to use a cloud database provider. 

### Fourth Deliverable: Time Series Modeling III and Website

We incorporate a machine learning approach to predict short-term (1day) stock movements for all IBEX35 companies. The model uses micro-level features, and a Random Forest classifier is trained to output both a prediction (buy or sell) and an associated probability. This simple model will serve as baseline ML approach.

Predictions, along with a general explanation of the methodology, model selection, and backtesting results, are available on our
[website](https://alexhayadela.github.io/10tothe6_TFG_2025_AlexDeLaHaya/docs/).

#### Website predictions
![image](imgs/preds.png)

The website is hosted using Github Pages. While the site is static, predictions are updated dynamically through an automation which runs the model, generates new forecasts and updates published content.

### Fifth Deliverable (Part A): Trading Bot
A bot will perform trades based on our model predictions. Connects to a broker API. WHen markets open executes predctions, when market is about to close it sells positions in which we have made a profit. Bot runs everyday when the market is open. By default it runs on a paper account with 100.000€.   


----

## Project Structure

```
/10tothe6_TFG_2025_AlexDeLaHaya
├── .github/workflows           # Automation scheduling and execution
|
├── data/                       # Stored OHLCV market data
├── imgs/ 
|
├── ingest/                     # Market data ingestion pipelines
├── models/                     # Financial time-series models
├── news/                       # News ingestion and newsletter automation
|
├── requirements/               # Python dependency specifications
├── .gitignore                  # Git ignore rules
└── README.md                   # Project overview and usage
```
----
## Project Setup

Open a terminal console and execute:
```bash
cd <your preferred projects root directory>
git clone https://github.com/alexhayadela/10tothe6_TFG_2025_AlexDeLaHaya.git
```
> [!IMPORTANT]
> Every command from now on, assumes you are in project root directory.

### Install python

Python 3.10+ is needed

### Install packages
Creating up a virtual environment is recommended to isolate the project dependencies.

```bash
python -m venv venv
```

Activate the environment:
```bash
venv\Scripts\activate
```

Install all the packages listed in `requirements.txt` with:
```bash
pip install -r requirements/all.txt
```

### Create .env file 

You need to create a .env file (project root) containing the following information:

1. EMAIL_USER=...@gmail.com
2. EMAIL_PASSWORD=...
3. GROQ_API_KEY=gsk_...
4. SUPABASE_API_KEY=sb_secret_...
5. SUPABASE_URL=https://<...>.supabase.co

- Gmail password must be a Google App Password. [How do I get one?](https://support.google.com/accounts/answer/185833).  
- Generate a Groq API key [here](https://console.groq.com/keys).
- Configure Supabase URL/API key in their [page](https://supabase.com). 

### Add Github Secrets

You need to configure github secrets to run the automation.

1. Go to Github → Settings → Secrets and variables → Actions → New repository secret
2. Add these secrets with the same values as your .env file: EMAIL_USER, EMAIL_PASSWORD, GROQ_API_KEY, SUPABASE_API_KEY, SUPABASE_URL

Launch web locally (development)
2. Can access web through internet or locally with (python -m http.server 8000, Ctrl+Shift+R to reload changes)

### Supabase db universe
Create your project. Instructions for manually creating tables.

TABLE news
id int8
date date
title text
section text
body text
url text
category text
relevance float8
sentiment text
id primary, url unique

TABLE news_entities
news_id int8
ticker text
news_id primary/fk REFERENCES id from news

TABLE newsletter
id int8
created_at timestamp
email text
id primary

TABLE ohlcv
ticker text
date date
open float8
hight float8
low float8
close float8
volume float8
primary composite ticker date
TABLE predictions
id int8
ticker text
date date
pred bool
proba float8
model text
primary id
All values except pk can be nullable.

Once you have created the database you must run "1st_run.ipynb". The first block will initialize your sqlite db, and the output of the second needs to be uploaded to your supabase project. 

There may arrive some time where you hit the free tier limit (500mb) migrate then erase but keep 250 last rows for ohlcv.

### Ml learning models

You can run ml learning models , inside models/train.py modify predictions.py accordingly to use the best model or the one that most suits you. (Default model:..)

Keep in mind that to have the latest market/news data available locally do python -m db.migrations. Run your mlmmodel afterwards.


----
### Activate trading bot
IB settings
Install the latest [IB Gateway](https://www.interactivebrokers.com/es/trading/ibgateway-latest.php) version.

You need to leave the program open for the bot to work. Change port if you want to swap from paper to real account (by default paper)
Enable API calls

```bash 
    //run as admin or "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
    .\trading\bot.ps1
```

If you ever want to stop bot 
##### How to erase tasks 
Unregister-ScheduledTask -TaskName "Open Positions" -Confirm:$false
Unregister-ScheduledTask -TaskName "Close Positions" -Confirm:$false

### Web development
In order to modify the website update docs/ with your own logic.
To run locally 
```bash
python -m http.server 8000
```
Then in browser type http://localhost:8000/docs/. To refresh changes press ctrl+shift+r.


## Project Usage
You can continue developing this project by forking this repository. If you only want you see web for daily predicitons use link and contact me so i add you to my personal newsletter. 

>[!CAUTION]
> Use this information at your own risk. ... Past returns dont guarantee ... El rendimiento pasado no es indicativo de resultados futuros. Las inversiones implican riesgos, incluida la posible pérdida del capital invertido. Las predicciones son estimaciones y no constituyen asesoramiento financiero. Actúa bajo tu propia responsabilidad.

## Conclusions
-> Read my final thesis for a more detailled explanations. Stock market produces a very noisy signal. With our models we are able to gain a bit of an edge (+50% accuracy). Two downside accuracy doesnt translate to money, e.g. days with more or less returns. And the key thing, commisions and slippage (buy / sell difference), with high frequency trading most of the profit is eaten by those. 
Also you are playing with money and it is hard to see losses, rather live stress free without having to worry much about things. I would suggest to invest in ETF's especially SP500 ones ETFs are a compound combination of assets. Are usually good because they diversify (reducing risk/variance), and provide fair returns without being involved. I express my concerns here about capitalism, wars and AI as the pilars of our economy are being held by weak .... Daily or weekly strategies are heavily influenced by the you can eat up to -30% if Trump decides to do nonsense. With the improvements of AI maybe human interaction isn't needed for companies to run. Agentic AI roles.. If peoples jobs are terminated how will economy run. Who will pay for products if noone has money. Is this why future is very uncertain.

focus less on money, wealth, and remember to spend quality to time with family and friends. 

Peace out, claude.opus.4.6~~Alex De La Haya Gutiérrez~~


## Author

Alex De La Haya Gutiérrez

