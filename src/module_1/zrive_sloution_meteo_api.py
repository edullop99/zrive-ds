
import matplotlib.pyplot as plt
import time
import requests
import logging
import json
import pandas as pd
from urllib.parse import urlencode
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)
logger.level = logging.INFO

API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

def _request_with_cooloff(
        url: str, headers: Dict[str, any], num_attempts: int, payload: Optional[Dict[str, any]] = None
):
    cooloff = 1
    for call_count in range(num_attempts):
        try: 
            if payload is None:
                response = requests.get(url, headers=headers)
            else: 
                response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response  # Asegúrate de devolver la respuesta aquí
        except requests.exceptions.ConnectionError as e: 
            logger.info("API refused the connection")
            logger.warning(e)
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2
                continue
            else: 
                raise
        except requests.exceptions.HTTPError as e: 
            logger.warning(e)
            if response.status_code == 404:
                raise
            logger.info(f"API return code {response.status_code} cooloff at {cooloff}")
            if call_count != (num_attempts - 1):
                time.sleep(cooloff)
                cooloff *= 2
                continue
            else:
                raise

def request_with_cooloff(
    url: str,
    headers: Dict[str, any],
    payload: Dict[str, any] = None, 
    num_attempts: int = 10,
) -> Dict[Any, Any]:
    return json.loads(
        _request_with_cooloff(
            url, 
            headers,
            num_attempts,
            payload,
        ).content.decode("utf-8")
    )

def get_data_meteo_api(longitude: float, latitude: float, start_date: str, end_date: str):
    headers = {}
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
        "timezone": "Europe/Madrid"
    }
    return request_with_cooloff(API_URL + urlencode(params, safe=","), headers)

def compute_monthly_statistics(data: pd.DataFrame, meteo_variables: List[str]):
    data['time'] = pd.to_datetime(data['time'])
    grouped = data.groupby([data['city'], data['time'].dt.to_period('M')])
    results = []

    for (city, month), group in grouped:
        monthly_stats = {"city": city, "month": month.to_timestamp()}   
        for variable in meteo_variables:
            monthly_stats[f"{variable}_max"] = group[variable].max()
            monthly_stats[f"{variable}_mean"] = group[variable].mean()
            monthly_stats[f"{variable}_min"] = group[variable].min()
            monthly_stats[f"{variable}_std"] = group[variable].std()
        results.append(monthly_stats)

    return pd.DataFrame(results)

def plot_timeseries(data: pd.DataFrame):
    rows = len(VARIABLES)
    cols = len(data["city"].unique())
    
    fig, axs = plt.subplots(rows, cols, figsize=(10, 6 * rows))

    for i, variable in enumerate(VARIABLES):
        for k, city in enumerate(data["city"].unique()):
            city_data = data[data["city"] == city]

            if city_data.empty:
                print(f"No data for {city} and {variable}")
                continue

            axs[i, k].plot(
                city_data["month"],
                city_data[f"{variable}_mean"],
                label=f"{city} (mean)",
                color=f"C{k}",
            )

            axs[i, k].fill_between(
                city_data["month"],
                city_data[f"{variable}_min"],
                city_data[f"{variable}_max"],
                alpha=0.2,
                color=f"C{k}",
            )

            axs[i, k].errorbar(
                city_data["month"],
                city_data[f"{variable}_mean"],
                city_data[f"{variable}_std"],
                fmt="none",
                alpha=0.5,
            )

            axs[i, k].set_xlabel("Date")
            axs[i, k].set_title(variable)
            if k == 0:
                axs[i, k].set_ylabel("Value")
            axs[i, k].legend()

    plt.tight_layout()
    plt.savefig("src/module_1/climate_evolution.png", bbox_inches="tight")
    plt.show()

def main():
    data_list = []
    start_date = "2010-01-01"
    end_date = "2020-12-31"

    for city, coordinates in COORDINATES.items():
        latitude = coordinates["latitude"]
        longitude = coordinates["longitude"]
        data = pd.DataFrame(get_data_meteo_api(longitude, latitude, start_date, end_date)["daily"]).assign(city=city)
        data_list.append(data)

    data = pd.concat(data_list)
    print(data.head())  # Muestra las primeras filas para verificar los datos

    calculated_ts = compute_monthly_statistics(data, VARIABLES)
    print(calculated_ts.head())  # Muestra las estadísticas mensuales

    plot_timeseries(calculated_ts)

if __name__ == "__main__":
    main()
