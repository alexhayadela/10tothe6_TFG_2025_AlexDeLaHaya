# 10**6 - Final Degree Thesis

This repository contains the code for my final degree thesis. The goal is to build a tool that empowers investors to obtain powerful insights at a glance. By leveraging multimodal, data-driven analysis, the system aims to support smarter investment decisions and potentially achieve better returns compared to traditional techniques.

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
```
cd <your preferred projects root directory>
git clone https://github.com/alexhayadela/10tothe6_TFG_2025_AlexDeLaHaya.git

```
### Install python

Python 3.10+ is needed

### Install virtualenv
Setting up a virtualenv is recommended to isolate the project dependencies from other Python projects on your machine.
It allows you to manage packages on a per-project basis, avoiding potential conflicts between different projects.

In the project root directory execute:
```bash
pip3 install virtualenv
virtualenv --version
```

### Prepare virtualenv for the project
In the root of the project folder run to create a virtualenv named `venv`:
```bash
virtualenv venv
```

If you list the contents of the project root directory, you will see that it has created a new folder named `venv` that contains the virtualenv:
```bash
ls -l
```

The next step is to activate your new virtualenv for the project:
```bash
source venv/bin/activate
```

or for Windows...
```cmd
venv\Scripts\activate.bat
```

This will load the python virtualenv for the project.

### Installing packages in your virtualenv
Make sure you are in the root of the project folder and that your virtualenv is activated (you should see `(venv)` in your terminal prompt).
And then install all the packages listed in `requirements.txt` with:
```bash
pip install -r requirements/all.txt
```

If you need to add more packages in the future, you can install them with pip and then update `requirements.txt` with:
```bash
pip freeze > requirements/all.txt
```

### Create .env file 

You need to create a .env file (project root) containing the following information:

1. EMAIL_USER=example@gmail.com
2. EMAIL_PASSWORD=asdf asdf adsf asdf
3. EMAIL_TARGET_USER=example2@gmail.com

The password must be an App Password from Google. ¿How to obtain an App Password for Gmail?: https://support.google.com/accounts/answer/185833

### Configure Github Secrets

You need to configure github secrets to run the automation.

1. Go to Github → Settings → Secrets and variables → Actions → New repository secret
2. Add these secrets with the same values as your .env file: EMAIL_USER, EMAIL_PASSWORD, EMAIL_TARGET_USER

Github Secrets guide: https://docs.github.com/en/actions/security-guides/encrypted-secrets

----
## Project description

### First Deliverable: Newsletter Automation

A daily newsletter is sent (automatically at 9.00 a.m UTC) containing the top-10 most relevant stock-related news of today.

The pipeline is the following. First fetch Expansión RSS (Mercados, Ahorro, Empresas). Then preprocess and filter relevant news. Convert to html, embed an image and finally send the newsletter. The workflow can be executed both locally and from GitHub Actions.

#### Newsletter example
![image](imgs/polite_newsletter.png)

### Second Deliverable: Database Ingestion and Time Series Modeling

From Monday to Friday after the Spanish market closes, Open, High, Low, Close, and Volume (OHLCV) data are automatically fetched for all IBEX35 companies and appended to the database.

We then apply ARIMA models to BBVA.MC prices and log returns over short-term and long-term horizons. Prices are non-stationary, while log returns behave as white noise, making ARIMA a suitable baseline model. Forecast variance increases over time, making long-term predictions unreliable. 

#### Arima log returns 
![image](imgs/arima_log_returns.png)


### Second Deliverable: Database Ingestion and Time Series Modeling
!!!!!!! ADD supabase db creation do before deliverables
----
## Author

Alex De La Haya Gutiérrez

