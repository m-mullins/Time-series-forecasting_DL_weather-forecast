from weather_data_downloader import WeatherDataDownloader

# Liste des stations étudiées [Montréal/PET,Montréal/St-Hubert,Montréal/Mirabel]
stations = [30165,48374,49608]
# stations = [49608]
years = list(range(2020,2022))
months = list(range(1, 13))

# Create an instance of WeatherDataDownloader and use it
downloader = WeatherDataDownloader(stations, years, months)
downloader.download_data_stations_years_months()

# Move CSVs to raw_csv folder
downloader.move_to_raw_csv()

print("done")