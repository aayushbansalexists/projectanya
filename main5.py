import streamlit as st
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import requests
import io
from streamlit.components.v1 import html
import math

st.title("Factory Site Selection with Custom Customer Distribution (US or India)")

# Country selection
country = st.selectbox("Select Country", ["US", "India"], index=0)

# Load dataset based on country
@st.cache_data
def load_cities(country):
    if country == "US":
        url = "https://simplemaps.com/static/data/us-cities/uscitiesv1.5.csv"
        try:
            r = requests.get(url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df = df[['city', 'state_name', 'lat', 'lng', 'population']]
            df = df[df['population'] >= 5000]
            return df
        except Exception as e:
            st.error(f"Error loading US cities dataset: {e}")
            return pd.DataFrame()
    elif country == "India":
        url = "https://raw.githubusercontent.com/geekmj/indian-cities/master/indian-cities.csv"
        try:
            r = requests.get(url)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            # Columns: name, state, population, latitude, longitude
            df = df.rename(columns={"name": "city", "state": "state_name", "latitude": "lat", "longitude": "lng"})
            df = df[['city', 'state_name', 'lat', 'lng', 'population']]
            df = df[df['population'] >= 50000]  # Filter for major cities
            return df
        except Exception as e:
            st.error(f"Error loading India cities dataset: {e}")
            return pd.DataFrame()

cities_df = load_cities(country)

if not cities_df.empty:
    st.subheader("Step 1: Select Cities and Specify Customer Counts")
    city_options = (cities_df['city'] + ", " + cities_df['state_name']).tolist()
    
    selected_cities = st.multiselect(
        f"Select {country} cities where you have customers",
        options=city_options,
        default=city_options[:5],  # Pre-select first 5 for demo
        help=f"Choose from real {country} cities. Then specify customer counts below."
    )
    
    # For each selected city, provide a number input for customer count
    customer_data = []
    for city_state in selected_cities:
        city, state = [x.strip() for x in city_state.split(",")]
        count = st.number_input(
            f"Number of customers in {city}, {state}",
            min_value=0,
            value=1000,  # Default value
            step=100,
            help="Enter the approximate number of customers here."
        )
        # Find lat/lon from dataset
        match = cities_df[(cities_df['city'] == city) & (cities_df['state_name'] == state)]
        if not match.empty:
            row = match.iloc[0]
            customer_data.append({
                'city': city,
                'state_name': state,
                'lat': row['lat'],
                'lng': row['lng'],
                'customers': count
            })
    
    user_customers_df = pd.DataFrame(customer_data)
    st.dataframe(user_customers_df)
else:
    st.warning(f"No {country} cities data loaded. Cannot proceed with customer selection.")

# Function to calculate weighted geographic centroid
def weighted_geographic_centroid(customers_df):
    if customers_df.empty:
        return None, None
    
    x = 0
    y = 0
    z = 0
    total_weight = 0

    for _, row in customers_df.iterrows():
        lat_rad = math.radians(row['lat'])
        lon_rad = math.radians(row['lng'])
        weight = row['customers']

        x += weight * math.cos(lat_rad) * math.cos(lon_rad)
        y += weight * math.cos(lat_rad) * math.sin(lon_rad)
        z += weight * math.sin(lat_rad)
        total_weight += weight

    if total_weight == 0:
        return None, None

    x /= total_weight
    y /= total_weight
    z /= total_weight

    lon = math.atan2(y, x)
    hyp = math.sqrt(x * x + y * y)
    lat = math.atan2(z, hyp)

    return math.degrees(lat), math.degrees(lon)

# Input for potential factory sites
st.subheader("Step 2: Input Potential Factory Sites (Optional)")
sites_input = st.text_area(
    "Enter sites (one per line as: name,lat,lon). Leave empty to compute optimal location via weighted centroid.",
    "Site A,40.7128,-74.0060\nSite B,34.0522,-118.2437\nSite C,41.8781,-87.6298" if country == "US" else "Site A,28.7041,77.1025\nSite B,19.0760,72.8777\nSite C,12.9716,77.5946",
    height=150
)

potential_sites = []
if sites_input.strip():
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

# Scoring function: Proximity to customers, weighted by customer counts
def compute_customer_proximity_score(site_lat, site_lon, customers_df):
    if customers_df.empty:
        return 0.0
    dists = []
    weights = []
    for _, row in customers_df.iterrows():
        dist = geodesic((site_lat, site_lon), (row['lat'], row['lng'])).km
        dists.append(dist)
        weights.append(row['customers'])  # Weight by user-specified customer count
    weighted_avg_dist = np.average(dists, weights=weights)
    return 1 / (1 + weighted_avg_dist / 1000)  # Normalize to ~[0,1]; adjust divisor for scale

# Calculate and find best site
if st.button("Calculate Ideal Location") and not user_customers_df.empty:
    if potential_sites:
        # If sites provided, score them
        best_site = None
        best_score = -np.inf
        for site in potential_sites:
            score = compute_customer_proximity_score(site['lat'], site['lon'], user_customers_df)
            site['score'] = score
            if score > best_score:
                best_score = score
                best_site = site
        
        st.success(f"Ideal location among provided sites: {best_site['name']} at ({best_site['lat']:.4f}, {best_site['lon']:.4f}) with proximity score {best_score:.4f}")
        
        # Also compute centroid for comparison
        centroid_lat, centroid_lon = weighted_geographic_centroid(user_customers_df)
        if centroid_lat is not None:
            centroid_score = compute_customer_proximity_score(centroid_lat, centroid_lon, user_customers_df)
            st.info(f"For comparison, weighted centroid location: ({centroid_lat:.4f}, {centroid_lon:.4f}) with score {centroid_score:.4f}")
        
        map_center = [best_site['lat'], best_site['lon']]
    else:
        # If no sites provided, compute weighted centroid as the ideal location
        centroid_lat, centroid_lon = weighted_geographic_centroid(user_customers_df)
        if centroid_lat is not None:
            centroid_score = compute_customer_proximity_score(centroid_lat, centroid_lon, user_customers_df)
            st.success(f"Computed ideal location (weighted centroid): ({centroid_lat:.4f}, {centroid_lon:.4f}) with proximity score {centroid_score:.4f}")
            map_center = [centroid_lat, centroid_lon]
        else:
            st.error("Unable to compute centroid - no customer data with weights.")
            map_center = [39.8283, -98.5795] if country == "US" else [20.5937, 78.9629]  # Default centers
    
    # Create interactive map
    m = folium.Map(location=map_center, zoom_start=4 if country == "US" else 5)
    
    # Add markers for customer cities (scaled by customer count for radius)
    for _, row in user_customers_df.iterrows():
        radius = max(3, min(20, row['customers'] / 1000))  # Scale radius visually
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"{row['city']}, {row['state_name']} (Customers: {row['customers']})"
        ).add_to(m)
    
    if potential_sites:
        # Add markers for provided sites
        for site in potential_sites:
            color = 'red' if 'score' in site and site['score'] == best_score else 'gray'
            folium.Marker(
                location=[site['lat'], site['lon']],
                popup=f"{site['name']}: Score {site['score']:.4f}",
                icon=folium.Icon(color=color)
            ).add_to(m)
    
    if not potential_sites or centroid_lat is not None:
        # Add marker for centroid
        folium.Marker(
            location=[centroid_lat, centroid_lon],
            popup="Weighted Centroid",
            icon=folium.Icon(color='green', icon='star')
        ).add_to(m)
    
    # Render map using HTML to avoid JSON serialization error
    map_html = m._repr_html_()
    html(map_html, width=800, height=500)
else:
    st.info("Select country, cities, specify customer counts, and (optionally) input sites. Then click the button to calculate.")
