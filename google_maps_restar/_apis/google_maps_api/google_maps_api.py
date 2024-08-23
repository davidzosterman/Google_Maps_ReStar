import requests
from google_maps_api.config import gm_place_api_url

def __call_api(endpoint: str):
    response = requests.get(gm_place_api_url + endpoint)
    print(f'Response status code: {response.status_code}')
    return response.json()

def find_place_from_text(API_KEY: str, restaurant_name: str):
    endpoint = (f'/findplacefromtext/json'
       f'?fields=formatted_address%2Cname%2Crating%2Cplace_id'
       f'&input={restaurant_name}'
       f'&inputtype=textquery'
       f'&key={API_KEY}'
      )
    response = __call_api(endpoint)
    return response

def place_details(API_KEY: str, place_id: str):
    endpoint = (f'/details/json'
       f'?fields=reviews'
       f'&place_id={place_id}'
       f'&key={API_KEY}'
      )
    response = __call_api(endpoint)
    return response
