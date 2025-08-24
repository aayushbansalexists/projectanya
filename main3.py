import streamlit as st
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import requests
import io
from streamlit.components.v1 import html

st.title("Factory Site Selection with Custom Customer Distribution")

# Load real US cities dataset as base for customer locations
@st.cache_data
def load_us_cities():
    url = "https://simplemaps.com/static/data/us-cities/uscitiesv1.5.csv"
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df[['city', 'state_name', 'lat', 'lng', 'population']]
        df = df[df['population'] >= 5000]
        return df
    except Exception as e:
        st.error(f"Error loading real US cities dataset: {e}")
        return pd.DataFrame()

us_cities = load_us_cities()

if not us_cities.empty:
    st.subheader("Step 1: Select Cities and Specify Customer Counts")
    # Create a list of city-state strings for multiselect
    city_options = (us_cities['city'] + ", " + us_cities['state_name']).tolist()
    
    selected_cities = st.multiselect(
        "Select cities where you have customers",
        options=city_options,
        default=city_options[:5],  # Pre-select first 5 for demo
        help="Choose from real US cities. Then specify customer counts below."
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
        match = us_cities[(us_cities['city'] == city) & (us_cities['state_name'] == state)]
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
    st.warning("No US cities data loaded. Cannot proceed with customer selection.")

# Input for potential factory sites
st.subheader("Step 2: Input Potential Factory Sites (Optional)")
sites_input = st.text_area(
    "Enter sites (one per line as: name,lat,lon). Leave empty to auto-generate a US-wide grid.",
    "Site A,40.7128,-74.0060\nSite B,34.0522,-118.2437\nSite C,41.8781,-87.6298",
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
else:
    # Generate a grid of potential sites across continental US if none provided
    st.info("No sites provided. Generating a grid of ~200 candidate locations across the US.")
    min_lat, max_lat = 24.5, 49.5  # Approx continental US bounds
    min_lon, max_lon = -125.0, -66.5
    num_points = 200  # Total points; adjust for density
    lats = np.linspace(min_lat, max_lat, int(np.sqrt(num_points)))
    lons = np.linspace(min_lon, max_lon, int(np.sqrt(num_points)) + 1)
    idx = 1
    for lat in lats:
        for lon in lons:
            potential_sites.append({'name': f"GridSite{idx}", 'lat': lat, 'lon': lon})
            idx += 1
            if idx > num_points:
                break
        if idx > num_points:
            break

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
if st.button("Calculate Ideal Location") and potential_sites and not user_customers_df.empty:
    best_site = None
    best_score = -np.inf
    for site in potential_sites:
        score = compute_customer_proximity_score(site['lat'], site['lon'], user_customers_df)
        site['score'] = score
        if score > best_score:
            best_score = score
            best_site = site
    
    st.success(f"Ideal location: {best_site['name']} at ({best_site['lat']:.4f}, {best_site['lon']:.4f}) with proximity score {best_score:.4f}")
    
    # Create interactive map centered on best site
    m = folium.Map(location=[best_site['lat'], best_site['lon']], zoom_start=4)
    
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
    
    # Add markers for potential sites (limit to 200 for performance if grid is large)
    displayed_sites = potential_sites[:200] if len(potential_sites) > 200 else potential_sites
    for site in displayed_sites:
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
    st.info("Select cities, specify customer counts, and (optionally) input sites. Then click the button to calculate.")
