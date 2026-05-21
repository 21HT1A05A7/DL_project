import redis
import json

# CONNECT TO REDIS
r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

# LOAD JSON FILE
with open("food_data.json", "r") as file:
    data = json.load(file)

# STORE ENTIRE DATA UNDER ONE KEY
r.set('food_details', json.dumps(data))

print("All food data stored successfully under key: food_details")