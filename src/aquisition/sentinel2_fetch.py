import ee
import geemap


# Replace with your GCP Project ID
# PROJECT = "smart-vegetation" 
ee.Initialize() # project=PROJECT

print("Earth Engine initialized successfully 🚀")

def fetch_sentinel2(aio_coords, start_date, end_date, filename):
    #Define AOI (list of [xmin, ymin, xmax, ymax])
    aoi = ee.Geometry.Rectangle(aio_coords)
    
    # Load Sentinel-2
    sentinel = ee.Imagecollection("COPERNICUS/S2") \
                .filterBounds(aoi) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    
    #Select median image
    image = sentinel.median()
    
     # Visualization params
    vis_params = {
        "bands": ["B4", "B3", "B2"],  # RGB
        "min": 0,
        "max": 3000
    }
    
    #Export to GeoTIFF
    geemap.ee_export_image(
        image.clip(aoi),
        filename=filename,
        scale=10,
        region=aoi
    )
    print(f'Saved Sentinel-2 image to {filename}')

#Example: Lahore City
fetch_sentinel2([74.25, 31.45, 74.45, 31.65], "2024-01-01", "2024-01-32", "data/raw/lahore.tif")