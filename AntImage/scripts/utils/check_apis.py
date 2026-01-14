import requests

def check_inaturalist():
    print("Checking iNaturalist...")
    # Search for Thailand place ID
    url = "https://api.inaturalist.org/v1/places/autocomplete?q=Thailand"
    resp = requests.get(url)
    if resp.status_code == 200:
        results = resp.json()['results']
        if results:
            print(f"Found {len(results)} places for 'Thailand'. Top result: {results[0]['id']} - {results[0]['name']}")
            return results[0]['id']
    else:
        print(f"Error: {resp.status_code}")
    return None

def check_antweb():
    print("\nChecking AntWeb...")
    # Try to fetch one specimen from Thailand with headers
    url = "https://api.antweb.org/v3.1/specimens?country=Thailand&limit=5"
    headers = {
        "User-Agent": "AntImageBot/1.0 (Contact: myemail@example.com)"
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"AntWeb status: {resp.status_code}")
        if 'specimens' in data:
             print(f"Found {len(data['specimens'])} specimens.")
             if len(data['specimens']) > 0:
                 print(f"Sample: {data['specimens'][0].get('catalogNumber')}")
        else:
             print("No 'specimens' key in response or different format.")
             print(data.keys())
    else:
        print(f"Error: {resp.status_code}")
        print(resp.text[:200])

if __name__ == "__main__":
    check_inaturalist()
    check_antweb()
