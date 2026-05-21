import redis
import json

# CONNECT TO REDIS
r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

# GET DATA FROM REDIS
data = r.get("food_details")

# CHECK IF EXISTS
if data is None:
    print("food_details not found")

else:
    # Convert JSON string back to Python dictionary
    food_data = json.loads(data)

    print(food_data)