import pandas as pd
import os
import shutil

class DataCleaner:

    def __init__(self, station):
        self.station = station
        self.csv_file_start = f"climate_hourly_{str(self.station)}_"
        
    def csvs_to_df(self):
        # Concat all raw csv files for a station 
        # Clean it by keeping only wanted columns and forward filling missing values

        # List all files in the raw_csv directory
        source_directory = "."
        target_directory = os.path.join(source_directory, "raw_csv")
        files = os.listdir(target_directory)
        
        # Sort files in numerical order
        sorted_files = sorted(files, key=lambda filename: (
            int(filename.split("_")[2]),
            int(filename.split("_")[3]),
            int(filename.split("_")[4].split(".")[0])
        ))
        files = sorted_files

        # Initialize dataframe list
        df_list = []

        # Loop through files and append files for required station
        for filename in files:
            if filename.startswith(self.csv_file_start) and filename.endswith(".csv"):
                csv_file_path = target_directory + "\\" + filename
                df = pd.read_csv(csv_file_path)
                df_list.append(df)
        df_station = pd.concat(df_list,ignore_index=True)
                
        # Keep wanted columns only
        df_station = df_station[["Date/Time (LST)","Temp (°C)","Rel Hum (%)","Precip. Amount (mm)","Stn Press (kPa)","Wind Spd (km/h)"]]
        df_station = df_station.rename(columns = {'Temp (°C)':'Temp (degC)'})
        print(f"\nStation {self.station} dataframe")
        print(df_station.head())

        # Check for missing data and fill missing data
        df_station.fillna(method='ffill', inplace=True)
        print(f"\nCheck for empty cells: {self.station}")
        print(df_station.isnull().any())

        # Send concatenated df to pickle and csv
        pkl_name = f"all_data_{str(self.station)}.pkl"
        df_station.to_pickle(pkl_name)
        df_station.to_csv(f"all_data_{str(self.station)}.csv")
        source_path = os.path.join(".", pkl_name)
        target_path = os.path.join(".","pickles", pkl_name)
        shutil.move(source_path, target_path)

    def send_clean_files_to_folder(self,file_type,folder_name):
        # Sends cleaned up files (csv or pickle) to their folder

        files = os.listdir(".")
        for file in files:
            if file[-4:] == ".csv":
                source_path = os.path.join(".", file)
                target_path = os.path.join(".",folder_name, file)
                shutil.move(source_path, target_path)

    def fill_missing_col(self,stations,filled_stations,missing_station,column_name):
        # Fill missing data in a column with average data from 2 other stations

        # Read pickles
        df_list = []
        files = os.listdir(os.path.join(".", "pickles"))
        for file in files:
            target_file = ".\\pickles\\" + file
            df = pd.read_pickle(target_file)
            df_list.append(df)

        # Calculate the average of df1 and df2 for each row
        average_values = (df_list[filled_stations[0]][column_name] + df_list[filled_stations[1]][column_name]) / 2

        # Fill df3 with the calculated average values
        df_list[missing_station][column_name] = average_values

        # Check for empty cells
        print(f"\nStation {stations[missing_station]} dataframe")
        print(df_list[missing_station].head())
        print(f"\nCheck for empty cells: {stations[missing_station]}")
        print(df_list[missing_station].isnull().any())

        # Update modified pickle file
        pkl_name = f"all_data_{str(stations[missing_station])}.pkl"
        csv_name = f"all_data_{str(stations[missing_station])}.csv"
        df_new = df_list[missing_station]
        df_new.to_pickle(pkl_name)
        df_new.to_csv(csv_name)
        source_path = os.path.join(".", pkl_name)
        target_path = os.path.join(".","pickles", pkl_name)
        shutil.move(source_path, target_path)

    def check_missing_col(self,stations):
        # Check for missing data in columns

        # Read pickles
        df_list = []
        files = os.listdir(os.path.join(".", "pickles"))
        for file in files:
            target_file = ".\\pickles\\" + file
            df = pd.read_pickle(target_file)
            df_list.append(df)

        # Check for empty cells
        for station in range(len(stations)):
            print(f"\nStation {stations[station]} dataframe")
            print(df_list[station].head())
            print(f"\nCheck for empty cells: {stations[station]}")
            print(df_list[station].isnull().any())


# Run this script to clean data and concat to clean_csv and pkl
# Concat and clean data
stations = [30165,48374,49608]
for station in stations:
    clean_data = DataCleaner(station)
    clean_data.csvs_to_df()

# Check for empty columns
clean_data.check_missing_col(stations)

# Fill missing data with average data in 2 other stations
clean_data.fill_missing_col(stations,[0,1],2,"Precip. Amount (mm)")
clean_data.send_clean_files_to_folder("csv","clean_csv")

print("\ndone")