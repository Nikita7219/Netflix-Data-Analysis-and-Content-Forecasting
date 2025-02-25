from prophet import Prophet
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cleaned_netflix_data.csv")

# Remove rows where 'year' is 0
df = df[df['Year'] != 0]

# Convert year to datetime format
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Prepare data for forecasting
def train_prophet(data, column_name):
    df_prophet = data[['Year', column_name]].rename(columns={'Year': 'ds', column_name: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    return forecast

# Forecast for Total Releases, Movies, and TV Shows
forecast_total = train_prophet(df, 'total_releases')
forecast_movies = train_prophet(df, 'num_movies')
forecast_tv = train_prophet(df, 'num_tv_shows')

# Save forecasts
forecast_total[['ds', 'yhat']].to_csv("forecast_total.csv", index=False)
forecast_movies[['ds', 'yhat']].to_csv("forecast_movies.csv", index=False)
forecast_tv[['ds', 'yhat']].to_csv("forecast_tv.csv", index=False)
