import pandas as pd
import yfinance as yf


def get_ibex_tickers():
    
    tickers = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC",
    "PUIG.MC", "RED.MC", "SAB.MC", "SAN.MC", "TEF.MC", "UNI.MC"
    ]
    return tickers


def get_ibex_tickers_name():
    tickers = [
    'ACS, Actividades de Construcción y Servicios, S.A.',
    'Acerinox, S.A.',
    'Aena S.M.E., S.A.',
    'Amadeus IT Group, S.A.',
    'Acciona, S.A.',
    'Corporación Acciona Energías Renovables, S.A.',
    'Banco Bilbao Vizcaya Argentaria, S.A.',
    'Bankinter, S.A.',
    'CaixaBank, S.A.',
    'Cellnex Telecom, S.A.',
    'Inmobiliaria Colonial, SOCIMI, S.A.',
    'Endesa, S.A.',
    'Enagás, S.A.',
    'Fluidra, S.A.',
    'Ferrovial SE',
    'Grifols, S.A.',
    'International Consolidated Airlines Group S.A.',
    'Iberdrola, S.A.',
    'Industria de Diseño Textil, S.A.',
    'Logista Integral, S.A.',
    'Mapfre, S.A.',
    'MERLIN Properties SOCIMI, S.A.',
    'ArcelorMittal S.A.',
    'Naturgy Energy Group, S.A.',
    'Puig Brands SA',
    'Redeia Corporación, S.A.',
    'Banco de Sabadell, S.A.',
    'Banco Santander, S.A.',
    'Telefónica, S.A.',
    'Unicaja Banco, S.A.'
    ]
    return tickers

def get_macro_tickers():
    # Consider adding VIBEX, VSTOXX, VIX (30 day - volatility expectation)
    tickers = ["^IBEX", "^STOXX50E", "^GSPC"]
    return tickers

def get_all_tickers():
    ibex = get_ibex_tickers()
    macro = get_macro_tickers()
    return ibex + macro

def download_ticker(ticker: str, start=None, safe=True):
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, auto_adjust=False)
    if df.empty:
        return df
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    df["ticker"] = ticker
    # Coarse correction: remove data from markets that haven't closed yet
    if safe and len(df) > 0:
        if df["date"].iloc[-1] >= pd.Timestamp.utcnow().strftime("%Y-%m-%d"):
            df = df.iloc[:-1]
    
    return df[["ticker", "date", "open", "high", "low", "close", "volume"]]

