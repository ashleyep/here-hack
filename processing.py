import pandas as pd
import os
import glob
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString
from shapely import wkt
from geopy.distance import great_circle, geodesic
from pathlib import Path

# from shapely.geometry import Point, Polygon
# import geopandas as gpd



def main():
    # Assuming "probe_data" is in the current directory; adjust if needed

    # Loop through folders named 0 to 14
    try_roadlinks()
    raw_data = pre_proc_test()
    gen_labels(raw_data)


    # Generate labels (+1 = point in roundabout, -1 = point not in roundabout)
    
    # loop through the columns for longitude and latitude
    # for each point, check if it is in the roundabout bounds
    
    
    # generate_features(raw_data)

    
    # generate feature vecto

def generate_features(raw_data):

    #TODO
    return


def try_roadlinks():
    road_links = pd.read_csv('hamburg_extra_layers/hamburg_road_links.csv')
    df_rl = pd.DataFrame(road_links)
    for index, rl in road_links.iterrows():
  
        geo_wkt = rl['wkt_geom']
        geo = wkt.loads(geo_wkt)
        if isinstance(geo, LineString):
            start_point = geo.coords[0]
            end_point = geo.coords[-1]
            
            # Calculate geodesic distance between start and end points
            distance = geodesic(start_point[::-1], end_point[::-1]).meters
            
            # If distance is less than a threshold (e.g., 20 meters), it could be a roundabout
            if distance < 5 and rl['highway'] is not 'primary':
                print('roundabout!')
                print(geo)
                print(rl['highway'])
    



''''Generate labels for the data'''
def gen_labels(raw_data):
    roundabouts = pd.read_csv('hamburg_extra_layers/hamburg_rounsabouts.csv')
    longitude_latitude_pairs = list(zip(roundabouts['longitude'], roundabouts['latitude']))
    known_roundabout = (53.546567900000184, 9.957687000000005)
    # for each point in the raw data, check if it is in the roundabout bounds
    # if it is, label it as +1
    # otherwise, label it as -1
    labels = []

    radius = 13
    for index, row in raw_data.iterrows():
        point = (row['longitude'], row['latitude'])
        distance = geodesic(point, known_roundabout).meters
          #  dist = np.sqrt((point[0] - known_roundabout[0]) ** 2 + (point[1] - known_roundabout[1]) ** 2) * 111320  # Approximate meters
        if distance < radius:
            labels.append(1)
        else:
            labels.append(-1)
    
    print(labels)
        
    



def pre_proc():
    parent_dir = 'probe_data'

    # Create an empty list to hold dataframes
    df_list = []

    # Walk through all folders and files in the parent directory
    for foldername, subfolders, filenames in os.walk(parent_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                # Construct the full path to the CSV file
                file_path = os.path.join(foldername, filename)
                
                # Read the CSV file into a DataFrame and append it to the list
                df = pd.read_csv(file_path)
                df_list.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(df_list, ignore_index=True)
    del(combined_df['Unnamed: 0'])
    return combined_df
    
def pre_proc_test():  
    path = Path('probe_data/0')
    df_list = []
    for file in path.iterdir():
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    del(combined_df['Unnamed: 0'])
    
    print("Finished pre proc")
    return combined_df
        

def cluster_test(data):
    
    # Convert sampledate to datetime
    data['sampledate'] = pd.to_datetime(data['sampledate'])

    # Calculate time differences in seconds (optional, remove if not needed)
    data['time_diff'] = data['sampledate'].diff().dt.total_seconds().fillna(0)

    # Known roundabout location
    known_roundabout = (53.546567900000184, 9.957687000000005)  # (latitude, longitude)

    # Calculate distance from known roundabout using vectorized approach
    lat = data['latitude'].values
    lon = data['longitude'].values
    data['distance_to_roundabout'] = np.sqrt((lat - known_roundabout[0]) ** 2 + (lon - known_roundabout[1]) ** 2) * 111320  # Approximate meters

    # Filter data within a specific radius (e.g., 300 meters)
    radius = 300  # meters
    filtered_data = data[data['distance_to_roundabout'] <= radius]

    # Identify significant heading and speed changes using vectorized calculations
    heading_change = filtered_data['heading'].diff().abs().fillna(0)
    speed_change = filtered_data['speed'].diff().abs().fillna(0)

    # Create a composite feature for clustering
    filtered_data['turning_behavior'] = ((heading_change > 45) & (speed_change > 10)).astype(int)

    # Select relevant features for clustering
    features = filtered_data[['latitude', 'longitude','turning_behavior']].dropna()

    # Clustering using DBSCAN
    clustering = DBSCAN(eps=0.005, min_samples=5).fit(features)

    # Add cluster labels to the filtered data
    filtered_data['cluster'] = clustering.labels_

    # Display potential roundabout clusters
    potential_clusters = filtered_data[filtered_data['cluster'] != -1]
    print(potential_clusters[['latitude', 'longitude', 'cluster']])

# Example usage
# df = pd.read_csv('/path/to/your/probe_data.csv')  # Uncomment this line to read your data
# cluster_test(df)


if __name__ == "__main__":
    main()