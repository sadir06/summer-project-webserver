import threading

from hardware.state import HARDWARE_FIELDS

lock = threading.Lock()

cache = {
    "sun": None,
    "price": None,
    "demand": None,
    "deferables": None,
    "yesterday": None,
    "last_updated": None,
    **HARDWARE_FIELDS,
}
