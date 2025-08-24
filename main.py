import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from geopy.distance import geodesic
import json

st.title("Factory Location Optimizer")

# Helper function to parse location lists from text
def parse_locations(text):
    locations = []
    for line in text.strip().split('\n'):
        if line.strip():
            try:
                lat, lon = map(float, line.split(','))
                locations.append((lat, lon))
            except ValueError:
                st.error(f"Invalid location format: {line}")
    return locations

# Input for potential sites
st.subheader("Potential Factory Sites")
sites_input = st.text_area("Enter sites (one per line as: name,lat,lon,cost,weather_score,water_level_score,soil_quality_score,land_area)", 
                           "Site A,40.7128,-74.0060,1000000,0.8,0.7,0.9,500\nSite B,34.0522,-118.2437,800000,0.9,0.6,0.8,600\nSite C,41.8781,-87.6298,1200000,0.7,0.8,0.7,450",
                           height=150)
potential_sites = []
for line in sites_input.strip().split('\n'):
    if line.strip():
        parts = line.split(',')
        if len(parts) == 8:
            name, lat, lon, cost, weather, water, soil, land = parts
            potential_sites.append({
                'name': name.strip(),
                'lat': float(lat),
                'lon': float(lon),
                'cost': float(cost),
                'weather': float(weather),
                'water_level': float(water),
                'soil_quality': float(soil),
                'land_area': float(land)
            })
        else:
            st.error(f"Invalid site format: {line}")

# Inputs for locations
st.subheader("Key Locations")
customers_input = st.text_area("Customer locations (one per line as: lat,lon)", "40.7306,-73.9352\n34.0736,-118.2923", height=100)
raw_materials_input = st.text_area("Raw material sources (one per line as: lat,lon)", "41.2565,-95.9345", height=100)
ports_input = st.text_area("Ports (one per line as: lat,lon)", "33.7456,-118.2617", height=100)
transport_input = st.text_area("Major transport hubs (one per line as: lat,lon)", "41.9786,-87.9048", height=100)

customers = parse_locations(customers_input)
raw_materials = parse_locations(raw_materials_input)
ports = parse_locations(ports_input)
transport_hubs = parse_locations(transport_input)

# Custom weights
st.subheader("Parameter Weights (0-1)")
w_customer_dist = st.slider("Customer distance weight", 0.0, 1.0, 0.2)
w_raw_dist = st.slider("Raw material distance weight", 0.0, 1.0, 0.15)
w_port_dist = st.slider("Port distance weight", 0.0, 1.0, 0.1)
w_transport_dist = st.slider("Transport distance weight", 0.0, 1.0, 0.1)
w_cost = st.slider("Cost weight (lower is better)", 0.0, 1.0, 0.15)
w_weather = st.slider("Weather weight", 0.0, 1.0, 0.05)
w_water = st.slider("Water level weight", 0.0, 1.0, 0.05)
w_soil = st.slider("Soil quality weight", 0.0, 1.0, 0.05)
w_land = st.slider("Land area weight (higher is better)", 0.0, 1.0, 0.1)

parameters_weights = {
    'customer_distance': w_customer_dist,
    'raw_material_distance': w_raw_dist,
    'port_distance': w_port_dist,
    'transport_distance': w_transport_dist,
    'cost': w_cost,
    'weather': w_weather,
    'water_level': w_water,
    'soil_quality': w_soil,
    'land_area': w_land
}

# Scoring functions
def compute_distance_score(site_lat, site_lon, locations):
    if not locations:
        return 0
    dists = [geodesic((site_lat, site_lon), loc).km for loc in locations]
    avg_dist = np.mean(dists)
    return 1 / (1 + avg_dist / 100)  # Scale to make score 0-1-ish; adjust divisor as needed

def score_site(site, max_cost, max_land):
    score = 0
    score += parameters_weights['customer_distance'] * compute_distance_score(site['lat'], site['lon'], customers)
    score += parameters_weights['raw_material_distance'] * compute_distance_score(site['lat'], site['lon'], raw_materials)
    score += parameters_weights['port_distance'] * compute_distance_score(site['lat'], site['lon'], ports)
    score += parameters_weights['transport_distance'] * compute_distance_score(site['lat'], site['lon'], transport_hubs)
    score += parameters_weights['cost'] * (1 - site['cost'] / max_cost) if max_cost > 0 else 0
    score += parameters_weights['weather'] * site['weather']
    score += parameters_weights['water_level'] * site['water_level']
    score += parameters_weights['soil_quality'] * site['soil_quality']
    score += parameters_weights['land_area'] * (site['land_area'] / max_land) if max_land > 0 else 0
    return score

# Compute best site
if st.button("Calculate Ideal Location") and potential_sites:
    max_cost = max(site['cost'] for site in potential_sites)
    max_land = max(site['land_area'] for site in potential_sites)
    best_site = max(potential_sites, key=lambda s: score_site(s, max_cost, max_land))
    best_score = score_site(best_site, max_cost, max_land)
    
    st.success(f"Ideal location: {best_site['name']} at ({best_site['lat']}, {best_site['lon']}) with score {best_score:.2f}")
    
    # Create map
    m = folium.Map(location=[best_site['lat'], best_site['lon']], zoom_start=5)
    
    # Add markers
    for loc in customers:
        folium.CircleMarker(loc, radius=5, color='blue', fill=True, popup='Customer').add_to(m)
    for loc in raw_materials:
        folium.CircleMarker(loc, radius=5, color='green', fill=True, popup='Raw Material').add_to(m)
    for loc in ports:
        folium.CircleMarker(loc, radius=5, color='purple', fill=True, popup='Port').add_to(m)
    for loc in transport_hubs:
        folium.CircleMarker(loc, radius=5, color='orange', fill=True, popup='Transport Hub').add_to(m)
    for site in potential_sites:
        color = 'red' if site == best_site else 'gray'
        folium.Marker([site['lat'], site['lon']], popup=site['name'], icon=folium.Icon(color=color)).add_to(m)
    
    st_folium(m, width=700, height=500)
else:
    st.info("Add sites and locations, then click the button to calculate.")
