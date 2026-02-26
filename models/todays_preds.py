import joblib

from models.utils import get_artifacts_path
from db.supabase.ingest_ohlcv import get_last_date
from db.ohlcv.queries import load_ohlcv_cloud
from db.utils_ohlcv import get_ibex_tickers
from models.trees.features import safe_build_features


model_type = "rf"
horizon = 1
name = f"{model_type}_h{horizon}_full.pkl"
model = joblib.load(get_artifacts_path() / name)

date = get_last_date()
tickers = get_ibex_tickers()
df_micro = load_ohlcv_cloud(tickers, date)



df = safe_build_features(df_micro, horizon)

preds = model.predict(df)
proba = model.predict_proba(df)[:, 1]




# i need 50/250 at least for each ticker to compute features