import threading
lock = threading.Lock()

cache = {
    "sun": None, 
    "price": None,
    "demand": None,
    "deferables": None,
    "yesterday": None,
    "last_updated": None,
}