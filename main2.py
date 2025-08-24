import streamlit as st
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import requests
import io
from streamlit.components.v1 import html  # Used to fix JSON serialization error

st.title("Factory Site Selection Using Real GIS Data")

# Load real US cities dataset as customer locations (from SimpleMaps, based on US Census/USGS data)
@st.cache_data
def load_us_cities():
    url = "https://simplemaps.com/static/data/us-cities/uscitiesv1.5.csv"
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Select relevant columns: city, state_name, lat, lng, population
        df = df[['city', 'state_name', 'lat', 'lng', 'population']]
        # Filter to cities with population >= 5000 for relevance and performance
        df = df[df['population'] >= 5000]
        return df
    except Exception as e:
        st.error(f"Error loading real US cities dataset: {e}")
        return pd.DataFrame()

us_cities = load_us_cities()

if not us_cities.empty:
    st.subheader("Sample of Loaded Real US Cities Data (as Customer Locations)")
    st.dataframe(us_cities.head(10))
    st.write(f"Total customer locations loaded: {len(us_cities)} (filtered to populations >= 5,000)")

# Input for potential factory sites
st.subheader("Input Potential Factory Sites")
sites_input = st.text_area(
    "Enter sites (one per line as: name,lat,lon)", 
    "Site A,40.7128,-74.0060\nSite B,34.0522,-118.2437\nSite C,41.8781,-87.6298",
    height=150
)

potential_sites = []
for line in sites_input.strip().split('\n'):
    if line.strip():
        parts = line.split(',')
        if len(parts) == 3:
            name, lat, lon = [p.strip() for p in parts]
            try:
                potential_sites.append({
                    'name': name,
                    'lat': float(lat),
                    'lon': float(lon)
                })
            except ValueError:
                st.error(f"Invalid lat/lon in: {line}")
        else:
            st.error(f"Invalid format in: {line}")

# Scoring function: Only one parameter - proximity to customers (weighted by population)
def compute_customer_proximity_score(site_lat, site_lon, cities_df):
    if cities_df.empty:
        return 0.0
    dists = []
    weights = []
    for _, row in cities_df.iterrows():
        dist = geodesic((site_lat, site_lon), (row['lat'], row['lng'])).km
        dists.append(dist)
        weights.append(row['population'])  # Weight by population for realism
    # Weighted average distance
    weighted_avg_dist = np.average(dists, weights=weights)
    # Score: Inverse of distance (higher score = better proximity), normalized to ~[0,1]
    return 1 / (1 + weighted_avg_dist / 1000)  # Divisor scales based on typical US distances

# Calculate and find best site
if st.button("Calculate Ideal Location") and potential_sites and not us_cities.empty:
    best_site = None
    best_score = -np.inf
    for site in potential_sites:
        score = compute_customer_proximity_score(site['lat'], site['lon'], us_cities)
        site['score'] = score
        if score > best_score:
            best_score = score
            best_site = site
    
    st.success(f"Ideal location: {best_site['name']} at ({best_site['lat']}, {best_site['lon']}) with proximity score {best_score:.4f}")
    
    # Create interactive map centered on best site
    m = folium.Map(location=[best_site['lat'], best_site['lon']], zoom_start=4)
    
    # Add markers for a sample of customer cities (limit to 200 for performance)
    sample_cities = us_cities.sample(min(200, len(us_cities)))
    for _, row in sample_cities.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"{row['city']}, {row['state_name']} (Pop: {row['population']})"
        ).add_to(m)
    
    # Add markers for potential sites
    for site in potential_sites:
        color = 'red' if site['name'] == best_site['name'] else 'gray'
        folium.Marker(
            location=[site['lat'], site['lon']],
            popup=f"{site['name']}: Score {site['score']:.4f}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Render map using HTML to avoid JSON serialization error
    map_html = m._repr_html_()
    html(map_html, width=800, height=500)
else:
    st.info("Enter potential sites and click the button to calculate using real US cities data.")
