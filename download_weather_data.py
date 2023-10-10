from weather_data_downloader import WeatherDataDownloader

# Liste des stations étudiées [Montréal/PET,Montréal/St-Hubert,Montréal/Mirabel]
# stations = [30165,48374,49608]
stations = [30165, 48374]
years = [2018, 2019]
months = list(range(5, 7))

# Create an instance of WeatherDataDownloader and use it
downloader = WeatherDataDownloader(stations, years, months)
downloader.download_data_stations_years_months()
print("Done")
