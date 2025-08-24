import streamlit as st
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from streamlit.components.v1 import html
import math
import requests
import json
import re

st.title("Factory Site Selection with Custom Customer Distribution - India Edition")

# Hardcoded path to the main cities CSV file
CSV_PATH = "india_cities.csv"

# Hardcoded path to the cost CSV file
COST_CSV_PATH = "cities_cleaned.csv"

# Load the main India cities dataset
try:
    india_cities = pd.read_csv(CSV_PATH)
    india_cities = india_cities[['city', 'state', 'population', 'latitude', 'longitude']]
    india_cities = india_cities.rename(columns={
        'state': 'state_name',
        'latitude': 'lat',
        'longitude': 'lng'
    })
    india_cities = india_cities[india_cities['population'] >= 50000].reset_index(drop=True)
except Exception as e:
    st.error(f"Error loading main CSV from {CSV_PATH}: {e}")
    india_cities = pd.DataFrame()

# Load the cost dataset
try:
    cost_df = pd.read_csv(COST_CSV_PATH)
    cost_df = cost_df[['city', 'cost_per_sqft', 'latitude', 'longitude']]
    cost_df = cost_df.rename(columns={'latitude': 'lat', 'longitude': 'lng'})
    # Calculate average cost for missing values
    average_cost = cost_df['cost_per_sqft'].mean()
    max_cost = cost_df['cost_per_sqft'].max()
    # Create a map for quick lookup (city to cost)
    cost_map = cost_df.set_index('city')['cost_per_sqft'].to_dict()
    # Also create a list of cost cities with lat/lon for nearest lookup
    cost_locations = cost_df[['city', 'lat', 'lng']].to_dict('records')
except Exception as e:
    st.error(f"Error loading cost CSV from {COST_CSV_PATH}: {e}")
    cost_df = pd.DataFrame()
    average_cost = 0
    max_cost = 1
    cost_map = {}
    cost_locations = []

# Initialize session state for customer data
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = []

# Function to find cost for a given lat/lon (nearest city in cost dataset, or average if not found)
def get_site_cost(site_lat, site_lon):
    if not cost_locations:
        return average_cost
    min_dist = float('inf')
    nearest_cost = average_cost
    for loc in cost_locations:
        dist = geodesic((site_lat, site_lon), (loc['lat'], loc['lng'])).km
        if dist < min_dist:
            min_dist = dist
            nearest_cost = cost_map.get(loc['city'], average_cost)
    return nearest_cost

# Function to update customer data from Gemini response
def update_customer_from_gemini(city, customers):
    match = india_cities[india_cities['city'].str.lower() == city.lower()]
    if not match.empty:
        row = match.iloc[0]
        for data in st.session_state.customer_data:
            if data['city'].lower() == city.lower():
                data['customers'] = customers
                return
        st.session_state.customer_data.append({
            'city': row['city'],
            'state_name': row['state_name'],
            'lat': row['lat'],
            'lng': row['lng'],
            'customers': customers
        })
    else:
        st.warning(f"City '{city}' not found in the dataset.")

if not india_cities.empty:
    st.subheader("Step 1: Specify Customer Distribution via Natural Language or Manual Selection")
    
    # Natural language input box for Gemini integration
    user_query = st.text_input("Tell me about your customers (e.g., 'I have 20000 customers in Mumbai')", "")
    if st.button("Process Query with Gemini") and user_query:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not set in Streamlit secrets. Add it in your app's settings on Streamlit Cloud.")
        else:
            try:
                # Query Gemini API
                gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
                headers = {
                    "Content-Type": "application/json"
                }
                data = {
                    "contents": [{
                        "parts": [{
                            "text": f"Parse this statement into a city name and customer count. Respond ONLY as valid JSON without any additional text, code blocks, or explanations: {{\"city\": \"CityName\", \"customers\": number}}. Do not include anything else. Statement: {user_query}"
                        }]
                    }]
                }
                response = requests.post(f"{gemini_url}?key={gemini_api_key}", headers=headers, json=data)
                if response.status_code == 200 and response.text:
                    gemini_result = response.json()
                    # Extract the generated text
                    gemini_response = gemini_result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                    if gemini_response:
                        # Extract the first well-formed { ... } block using regex
                        json_match = re.search(r'\{.*?\}', gemini_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            # Clean the extracted JSON: Remove extra garbage, replace single quotes
                            json_str = re.sub(r'``````|undefined|json|\s+', '', json_str).strip()
                            json_str = re.sub(r"'(?P<key>[^':]+)':", r'\"\g<key>\":', json_str)  # Replace single-quoted keys
                            json_str = json_str.replace("'", '"')  # Replace single-quoted values
                            parsed = json.loads(json_str)
                            city = parsed.get('city')
                            customers = int(parsed.get('customers', 0))
                            if city and customers > 0:
                                update_customer_from_gemini(city, customers)
                                st.success(f"Updated: {customers} customers in {city}")
                            else:
                                st.error("Could not parse city or customer count from response.")
                        else:
                            st.error("No valid JSON block found in response.")
                    else:
                        st.error("Gemini returned an empty response.")
                else:
                    st.error(f"Gemini server error: {response.status_code} - {response.text}")
            except json.JSONDecodeError as json_err:
                st.error(f"JSON parsing error: {json_err}. Raw response: {gemini_response}")
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    # Manual selection
    city_options = (india_cities['city'] + ", " + india_cities['state_name']).tolist()
    selected_cities = st.multiselect(
        "Manually select/add cities and edit counts below",
        options=city_options,
        help="Use this to manually adjust or add more."
    )
    
    # Display and edit customer data
    st.subheader("Current Customer Distribution")
    edited_data = []
    for data in st.session_state.customer_data:
        count = st.number_input(
            f"Customers in {data['city']}, {data['state_name']}",
            min_value=0,
            value=data['customers'],
            step=100,
            key=f"edit_{data['city']}"
        )
        data['customers'] = count
        if count > 0:
            edited_data.append(data)
    
    # Add any new manual selections
    for city_state in selected_cities:
        city, state = [x.strip() for x in city_state.split(",")]
        if not any(d['city'] == city for d in edited_data):
            match = india_cities[(india_cities['city'] == city) & (india_cities['state_name'] == state)]
            if not match.empty:
                row = match.iloc[0]
                count = st.number_input(
                    f"Number of customers in {city}, {state}",
                    min_value=0,
                    value=1000,
                    step=100
                )
                if count > 0:
                    edited_data.append({
                        'city': city,
                        'state_name': state,
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'customers': count
                    })
    
    st.session_state.customer_data = edited_data
    user_customers_df = pd.DataFrame(edited_data)
    st.dataframe(user_customers_df)
else:
    st.warning("CSV file not loaded. Ensure 'india_cities.csv' is in the app directory.")

# Slider for cost impact weight
cost_impact_weight = st.slider(
    "Impact of Cost per Sqft on Selection (0: none, 1: full)",
    0.0, 1.0, 0.5,
    help="Higher values prioritize lower cost locations more."
)

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
    "Site A,28.7041,77.1025\nSite B,19.0760,72.8777\nSite C,12.9716,77.5946",
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

# Scoring function incorporating cost
def compute_score(site_lat, site_lon, customers_df):
    if customers_df.empty:
        return 0.0
    
    # Distance score
    dists = []
    weights = []
    for _, row in customers_df.iterrows():
        dist = geodesic((site_lat, site_lon), (row['lat'], row['lng'])).km
        dists.append(dist)
        weights.append(row['customers'])
    weighted_avg_dist = np.average(dists, weights=weights)
    distance_score = 1 / (1 + weighted_avg_dist / 1000)  # Higher for closer
    
    # Cost score: Get cost for this site (nearest in cost dataset)
    site_cost = get_site_cost(site_lat, site_lon)
    cost_score = 1 - (site_cost / max_cost) if max_cost > 0 else 0  # Higher for lower cost
    
    # Combined score
    final_score = distance_score * (1 - cost_impact_weight) + cost_score * cost_impact_weight
    return final_score

# Calculate and find best site
if st.button("Calculate Ideal Location") and not india_cities.empty and not user_customers_df.empty:
    centroid_lat, centroid_lon = weighted_geographic_centroid(user_customers_df)
    centroid_score = None
    if centroid_lat is not None:
        centroid_score = compute_score(centroid_lat, centroid_lon, user_customers_df)
    
    if potential_sites:
        best_site = None
        best_score = -np.inf
        for site in potential_sites:
            score = compute_score(site['lat'], site['lon'], user_customers_df)
            site['score'] = score
            if score > best_score:
                best_score = score
                best_site = site
        
        st.success(f"Ideal location among provided sites: {best_site['name']} at ({best_site['lat']:.4f}, {best_site['lon']:.4f}) with score {best_score:.4f} (including cost impact)")
        
        if centroid_lat is not None:
            st.info(f"For comparison, weighted centroid location: ({centroid_lat:.4f}, {centroid_lon:.4f}) with score {centroid_score:.4f}")
        
        map_center = [best_site['lat'], best_site['lon']]
    else:
        if centroid_lat is not None:
            st.success(f"Computed ideal location (weighted centroid): ({centroid_lat:.4f}, {centroid_lon:.4f}) with score {centroid_score:.4f} (including cost impact)")
            map_center = [centroid_lat, centroid_lon]
        else:
            st.error("Unable to compute centroid - no customer data with weights.")
            map_center = [20.5937, 78.9629]
    
    m = folium.Map(location=map_center, zoom_start=5)
    
    for _, row in user_customers_df.iterrows():
        radius = max(3, min(20, row['customers'] / 1000))
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=radius,
            color='blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"{row['city']}, {row['state_name']} (Customers: {row['customers']})"
        ).add_to(m)
    
    for site in potential_sites:
        color = 'red' if 'score' in site and site['score'] == best_score else 'gray'
        folium.Marker(
            location=[site['lat'], site['lon']],
            popup=f"{site['name']}: Score {site['score']:.4f}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    if centroid_lat is not None:
        folium.Marker(
            location=[centroid_lat, centroid_lon],
            popup=f"Weighted Centroid: Score {centroid_score:.4f}",
            icon=folium.Icon(color='green', icon='star')
        ).add_to(m)
    
    map_html = m._repr_html_()
    html(map_html, width=800, height=500)
else:
    st.info("Add customer data via query or manual selection, and (optionally) input sites. Then click the button to calculate.")
