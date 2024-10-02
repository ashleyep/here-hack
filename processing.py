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
import geojson
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from haversine import haversine, Unit

# from shapely.geometry import Point, Polygon
# import geopandas as gpd



def main():
    # Assuming "probe_data" is in the current directory; adjust if needed

    # Loop through folders named 0 to 14
    try_roadlinks()
    centroid_yield = try_yields()
    df = pd.read_csv('centroid_coordinates.csv')
    # Prepare a list to collect new rows
    new_rows = []

    # Iterate through the centroids and prepare new rows
    for centroid in centroid_yield: 
        new_rows.append({'longitude': centroid[0], 'latitude': centroid[1]})

    # Create a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)

    # Concatenate the existing DataFrame with the new DataFrame
    df = pd.concat([df, new_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df.to_csv('centroid_coordinates.csv', index=False)



def convert_to_geojson(curve_list, output_file):    
    features = []
    print(curve_list)
    for geo in curve_list:
        geometry = geojson.LineString(geo.coords)
        feature = geojson.Feature(geometry=geometry)
        print(feature)
        features.append(feature)
    
    feature_collection = geojson.FeatureCollection(features)
    
    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)


def calculate_radius_of_curvature(p1, p2, p3):
    # Calculate the circumcenter (center of the circle through p1, p2, and p3)
    A = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    B = np.array([p1[0]**2 + p1[1]**2, p2[0]**2 + p2[1]**2, p3[0]**2 + p3[1]**2])
    try:
        center = np.linalg.solve(A, B)
        cx, cy = center[0]/2, center[1]/2
        
        # Calculate radius (distance from center to any point on the circle)
        radius = geodesic((cy, cx), (p1[1], p1[0])).meters
        return radius
    except np.linalg.LinAlgError:
        # If points are collinear, no circle can be formed
        return float('inf')  # Infinite radius means no curvature (straight line)

def try_roadlinks():
    road_links = pd.read_csv('hamburg_extra_layers/hamburg_road_links.csv')
    curve_list = []
    for index, rl in road_links.iterrows():
  
        geo_wkt = rl['wkt_geom']
        geo = wkt.loads(geo_wkt)
        
        if isinstance(geo, LineString):
            start_point = geo.coords[0]
            end_point = geo.coords[-1]
            coords = list(geo.coords)
            
            
            # Calculate geodesic distance between start and end points 
            distance = geodesic(start_point[::-1], end_point[::-1]).meters
           
            # If distance is less than a threshold (e.g., 20 meters), it could be a roundabout
            if distance < 10 and rl['highway'] != 'primary':
                for i in range(len(coords) - 2):
                        p1 = coords[i]
                        p2 = coords[i + 1]
                        p3 = coords[i + 2]
                        
                        radius = calculate_radius_of_curvature(p1, p2, p3)
                        if radius < 50:  # Lower radius means more curvature
                            # curved_linestrings.append(geo)
                            
                            centroid = geo.centroid.coords[0]
                            
                            curve_list.append((centroid[0], centroid[1]))
  
                            break  # Found a curve, move to the next LineString
    
        
    df = pd.DataFrame(curve_list, columns=['longitude', 'latitude'])
    print(df)
    
    # Save the DataFrame to a CSV file
    csv_file_path = 'centroid_coordinates.csv'
    df.to_csv(csv_file_path, index=False)
    # convert_to_geojson(curve_list = curve_list,output_file= 'road_links_12.geojson')
              
                

def try_yields():
    yield_signs = pd.read_csv('hamburg_extra_layers/hamburg_yield_signs.csv')
    df = pd.DataFrame(yield_signs)

    def haversine_distance(coord1, coord2):
        return haversine(coord1, coord2, unit=Unit.METERS)

    coordinates = list(zip(df['latitude'], df['longitude']))

    # DBSCAN clustering algorithm
    dbscan = DBSCAN(eps=35, min_samples=3, metric=haversine_distance)

    # Fit the model
    df['cluster'] = dbscan.fit_predict(coordinates)

    # Show the resulting clusters
    clusters = df[df['cluster'] != -1]  # Filter out noise (no cluster points)
    
    # Compute the central point (centroid) for each cluster
    centroids = clusters.groupby('cluster').agg({
        'longitude': 'mean',
        'latitude': 'mean'
    }).reset_index()

    # Output the centroids (central points) of each cluster
    centroid_list = centroids[['longitude', 'latitude']].values.tolist()


    return centroid_list

        
        



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



if __name__ == "__main__":
    main()