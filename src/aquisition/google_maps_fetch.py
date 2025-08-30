import requests


API_KEY="AIzaSyD67oPht5PjsZTADmxTenfB75_1Z2RbfyQ"
# client_Id=1053708136480-mfeeii94ieqhf0pjnufti7fbd6mcht48.apps.googleusercontent.com


# API_KEY = "YOUR_GOOGLE_API_KEY"
def fetch_map(lat, lng, zoom=15, size="640x640", filename="map.png"):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"
    r = requests.get(url)
    print(f"reponse: {r}")
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"Saved {filename}")

# Example: fetch Lahore city region
fetch_map(31.5820, 74.3294)