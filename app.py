import streamlit as st
import folium
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from streamlit.components.v1 import html
import math

st.title("Factory Site Selection with Custom Customer Distribution - India Edition")

# Hardcoded path to the CSV file
CSV_PATH = "india_cities.csv"

# Load the India cities dataset from the hardcoded CSV path
try:
    india_cities = pd.read_csv(CSV_PATH)
    # Select and rename relevant columns based on the provided CSV structure
    india_cities = india_cities[['city', 'state', 'population', 'latitude', 'longitude']]
    india_cities = india_cities.rename(columns={
        'state': 'state_name',
        'latitude': 'lat',
        'longitude': 'lng'
    })
    # Filter by population threshold (e.g., >= 50,000 for relevance)
    india_cities = india_cities[india_cities['population'] >= 50000].reset_index(drop=True)
except Exception as e:
    st.error(f"Error loading CSV from {CSV_PATH}: {e}")
    india_cities = pd.DataFrame()

# Initialize session state for customer data
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = []

# Function to update customer data from parsed response
def update_customer_from_parsed(city, customers):
    # Find the city in the dataset (case-insensitive match)
    match = india_cities[india_cities['city'].str.lower() == city.lower()]
    if not match.empty:
        row = match.iloc[0]
        # Check if city already exists in customer_data
        for data in st.session_state.customer_data:
            if data['city'].lower() == city.lower():
                data['customers'] = customers  # Update existing
                return
        # Add new if not found
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
    
    # Embed JavaScript for WebGPU-based model inference
    html("""
    <div id="webgpu-container">
        <input id="user-query" type="text" placeholder="Tell me about your customers (e.g., 'I have 20000 customers in Mumbai')" style="width: 100%; padding: 10px; margin-bottom: 10px;">
        <button id="process-query" style="padding: 10px; background-color: #4CAF50; color: white; border: none; cursor: pointer;">Process Query with WebGPU</button>
        <div id="status" style="margin-top: 10px;"></div>
    </div>
    <script type="text/javascript">
        (async () => {
            const userQueryInput = document.getElementById('user-query');
            const processButton = document.getElementById('process-query');
            const statusDiv = document.getElementById('status');

            // Check for WebGPU support
            if (!navigator.gpu) {
                statusDiv.innerHTML = 'WebGPU is not supported in this browser. Please use manual selection below.';
                return;
            }

            statusDiv.innerHTML = 'Initializing WebGPU model...';

            try {
                const { CreateMLCEngine } = await import('https://esm.run/@mlc-ai/web-llm');
                const engine = await CreateMLCEngine('Llama-3.2-1B-Instruct-q4f16_1-MLC', {
                    initProgressCallback: (report) => {
                        statusDiv.innerHTML = `${report.text} (${Math.round(report.progress * 100)}%)`;
                    }
                });

                statusDiv.innerHTML = 'Model loaded. Ready for queries.';

                processButton.addEventListener('click', async () => {
                    const query = userQueryInput.value.trim();
                    if (!query) return;

                    statusDiv.innerHTML = 'Processing query...';

                    try {
                        const prompt = `Parse this statement into a city name and customer count. Respond ONLY as valid JSON: {"city": "CityName", "customers": number}. Statement: ${query}`;
                        const response = await engine.chat.completions.create({
                            messages: [{ role: 'user', content: prompt }],
                            temperature: 0.7,
                            max_tokens: 100
                        });
                        const aiResponse = response.choices[0].message.content.trim();

                        // Automatically parse and send via postMessage
                        const parsed = JSON.parse(aiResponse);
                        const city = parsed.city;
                        const customers = parseInt(parsed.customers);
                        if (city && customers > 0) {
                            parent.postMessage({ type: 'update_customer', city: city, customers: customers }, '*');
                            statusDiv.innerHTML = `Processed: ${customers} customers in ${city}. Updating automatically...`;
                        } else {
                            statusDiv.innerHTML = 'Could not parse response. Use manual addition.';
                        }
                    } catch (error) {
                        statusDiv.innerHTML = `Error: ${error.message}. Use manual addition.`;
                    }
                });
            } catch (error) {
                statusDiv.innerHTML = `Failed to load model: ${error.message}. Use manual selection.`;
            }
        })();
    </script>
    """, height=200)

    # JavaScript listener for postMessage to update session state (run in another html to listen)
    html("""
    <script>
        window.addEventListener('message', function(event) {
            if (event.data.type === 'update_customer') {
                // Here, we can try to update a hidden input or localStorage
                localStorage.setItem('ai_city', event.data.city);
                localStorage.setItem('ai_customers', event.data.customers);
                // Trigger a rerun by simulating interaction if possible, but in Streamlit, we need Python to poll
            }
        });
    </script>
    """, height=1)

    # Poll localStorage in Python via JS (use st.javascript to read)
    ai_city = st.experimental_get_query_params().get('ai_city', [None])[0]  # Not direct
    # To read localStorage, use a hidden html with script that posts to a div
    # For simplicity, add a button to fetch from localStorage
    if st.button("Update from AI Processing"):
        # Use JS to get localStorage and display
        ai_city_js = st.javascript("localStorage.getItem('ai_city')")
        ai_customers_js = st.javascript("localStorage.getItem('ai_customers')")
        if ai_city_js and ai_customers_js:
            try:
                customers = int(ai_customers_js)
                update_customer_from_parsed(ai_city_js, customers)
                st.success(f"Added/Updated from AI: {customers} customers in {ai_city_js}")
                st.rerun()
            except ValueError:
                st.error("Invalid data from AI.")
        else:
            st.info("No AI data available. Process a query first.")

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

# Scoring function
def compute_customer_proximity_score(site_lat, site_lon, customers_df):
    if customers_df.empty:
        return 0.0
    dists = []
    weights = []
    for _, row in customers_df.iterrows():
        dist = geodesic((site_lat, site_lon), (row['lat'], row['lng'])).km
        dists.append(dist)
        weights.append(row['customers'])
    weighted_avg_dist = np.average(dists, weights=weights)
    return 1 / (1 + weighted_avg_dist / 1000)

# Calculate and find best site
if st.button("Calculate Ideal Location") and not india_cities.empty and len(st.session_state.customer_data) > 0:
    user_customers_df = pd.DataFrame(st.session_state.customer_data)
    if potential_sites:
        best_site = None
        best_score = -np.inf
        for site in potential_sites:
            score = compute_customer_proximity_score(site['lat'], site['lon'], user_customers_df)
            site['score'] = score
            if score > best_score:
                best_score = score
                best_site = site
        
        st.success(f"Ideal location among provided sites: {best_site['name']} at ({best_site['lat']:.4f}, {best_site['lon']:.4f}) with proximity score {best_score:.4f}")
        
        centroid_lat, centroid_lon = weighted_geographic_centroid(user_customers_df)
        if centroid_lat is not None:
            centroid_score = compute_customer_proximity_score(centroid_lat, centroid_lon, user_customers_df)
            st.info(f"For comparison, weighted centroid location: ({centroid_lat:.4f}, {centroid_lon:.4f}) with score {centroid_score:.4f}")
        
        map_center = [best_site['lat'], best_site['lon']]
    else:
        centroid_lat, centroid_lon = weighted_geographic_centroid(user_customers_df)
        if centroid_lat is not None:
            centroid_score = compute_customer_proximity_score(centroid_lat, centroid_lon, user_customers_df)
            st.success(f"Computed ideal location (weighted centroid): ({centroid_lat:.4f}, {centroid_lon:.4f}) with proximity score {centroid_score:.4f}")
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
            popup="Weighted Centroid",
            icon=folium.Icon(color='green', icon='star')
        ).add_to(m)
    
    map_html = m._repr_html_()
    html(map_html, width=800, height=500)
else:
    st.info("Add customer data via query or manual selection, and (optionally) input sites. Then click the button to calculate.")
