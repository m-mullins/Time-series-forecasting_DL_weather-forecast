import requests
import re

class WeatherDataDownloader:
    def __init__(self, stations, years, months):
        self.stations = stations
        self.years = years
        self.months = months

    def download_weather_data_month(self, url, save_path=None):
        # Bulk download weather data from the canadian climate weather website
        # Data is downloaded for 1 station for 1 month and with the hourly format

        try:
            # Send an HTTP GET request to the URL
            response = requests.get(url)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:

                # Extract the filename from the Content-Disposition header
                content_disposition = response.headers.get("Content-Disposition")
                if content_disposition:
                    file_name = re.findall('filename="(.+)"', content_disposition)[0]
                else:
                    file_name = "downloaded_file.csv"

                # Save the CSV file
                with open(save_path, "wb") as csv_file:
                    csv_file.write(response.content)
                print(f"File '{file_name}' downloaded successfully.")
            else:
                print(f"Failed to download the file. Status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def download_data_stations_years_months(self):
        # Download weather data for each station, year and month

        for station in self.stations:
            for year in self.years:
                for month in self.months:
                    csv_url = f"https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={station}&Year={year}&Month={month}&Day=14&timeframe=1&submit=Download+Data"
                    print(csv_url)
                    file_name = f"climate_hourly_{station}_{year}_{month}.csv"
                    self.download_weather_data_month(csv_url, file_name)

