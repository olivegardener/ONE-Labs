import requests

response = requests.get(
    "https://api.census.gov/data/2020/dec/pl",
    params={"get": "NAME", "for": "state:*", "key": "a3ebdf1648b7fb21df55df7246d9642f040c0ee0"}
)

print(response.json())
