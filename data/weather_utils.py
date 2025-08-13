import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEFAULT_CITY = os.getenv("DEFAULT_CITY")

def get_weather_forecast(city=None):
    """
    Fetches the weather forecast for the next 24 hours for a given city,
    calculates the average temperature, and maps it to 'hot' or 'cold' tags.
    If no city is provided, uses DEFAULT_CITY from environment variables.
    
    Returns a dict with:
      - avg_temp: float, average temperature in Celsius
      - weather_tag: 'hot' or 'cold'
      - condition: current weather condition string (e.g. "Clear", "Rain")
    """
    if not OPENWEATHER_API_KEY:
        raise EnvironmentError("OPENWEATHER_API_KEY not set in environment variables.")
    if not city:
        if not DEFAULT_CITY:
            raise EnvironmentError("DEFAULT_CITY not set in environment variables.")
        city = DEFAULT_CITY

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"OpenWeather API error: {response.status_code} {response.text}")

    data = response.json()

    if "list" not in data or not data["list"]:
        raise Exception("No forecast data found.")

    # 24h forecast = next 8 blocks of 3h each
    temps = [entry["main"]["temp"] for entry in data["list"][:8]]
    avg_temp = sum(temps) / len(temps)
    weather_tag = "hot" if avg_temp >= 20 else "cold"

    return {
        "avg_temp": round(avg_temp, 1),
        "weather_tag": weather_tag,
        "condition": data["list"][0]["weather"][0]["main"]
    }

def get_current_temperature(city=None):
    """
    Returns the average temperature in Celsius for the next 24 hours from the weather forecast.
    """
    forecast = get_weather_forecast(city)
    return forecast["avg_temp"]