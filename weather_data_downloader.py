import requests
import re
import os
import shutil

class WeatherDataDownloader:
    def __init__(self, stations, years, months):
        self.stations = stations
        self.years = years
        self.months = months

    def download_weather_data_month(self, url, save_path=None):
        # Bulk download weather data from the canadian climate weather website
        # Data is downloaded for 1 station for 1 month and with the hourly format
        # Regular download website: https://climate.weather.gc.ca/climate_data/daily_data_e.html?StationID=51157
        # Bulk download website: https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/


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
        print("CSVs downloaded succesfully")

    def move_to_raw_csv(self):

        # Define the source directory where the files are located
        source_directory = "."  # Current directory, change this to the appropriate path

        # Define the target directory (raw_csv subfolder)
        target_directory = os.path.join(source_directory, "raw_csv")

        # Ensure the target directory exists, create it if not
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        # List all files in the source directory
        files = os.listdir(source_directory)

        # Loop through the files and move those that meet the criteria
        for filename in files:
            if filename.startswith("climate_hourly") and filename.endswith(".csv"):
                source_path = os.path.join(source_directory, filename)
                target_path = os.path.join(target_directory, filename)
                shutil.move(source_path, target_path)
                print(f"Moved: {filename} to raw_csv")

        print("CSVs moved to raw_csv folder")


