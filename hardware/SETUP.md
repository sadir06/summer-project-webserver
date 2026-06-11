# Hardware network setup (phone hotspot + laptop + Picos)

## How sun data reaches the MPPT Pico

The MPPT Pico **pulls** data from your laptop; your laptop does not push to the Pico.

```
Azure game server
       ↓  (Flask poller fetches every ~5s)
Your laptop :8000/api/sun_data
       ↓  (MPPT Pico GETs this on its own schedule)
MPPT Pico firmware
```

1. Start Flask on the laptop: `python -m webserver.middleware.app`
2. Flask polls Azure and caches `sun` + `tick` in memory.
3. MPPT firmware calls `GET http://172.20.10.4:8000/api/sun_data` and reads JSON `{"sun": ..., "tick": ...}`.

`inference.py` is separate: it reads Azure + Pico telemetry directly. Flask is still required for the MPPT Pico.

## Public IP vs local hotspot IP

Your phone may show a **public** IP on the internet (e.g. `104.28.89.52`). That is **not** what the laptop or Picos use to talk to each other.

| IP type | Example | Used for |
|---------|---------|----------|
| Public (cellular/WAN) | `104.28.89.52` | Internet-facing only — **ignore for Pico/laptop config** |
| Local hotspot gateway | Often `172.20.10.1` or `192.168.x.1` | Wi-Fi gateway on your laptop |
| Laptop (static) | `172.20.10.4` | Flask + what MPPT Pico polls |
| Pico (DHCP) | e.g. `172.20.10.5`, `.8` | `PICO_URL_PV`, `PICO_URL_CP` in config |

After joining the hotspot, run `ipconfig` on the laptop and use the **Default Gateway** on the Wi-Fi adapter as `HOTSPOT_GATEWAY` in `hardware/config.py` if it is not `172.20.10.1`.

## Set laptop IP to 172.20.10.4 (Windows)

MPPT firmware hardcodes `SUN_SERVER_HOST = "172.20.10.4"`. When connected to the phone hotspot:

1. Settings → Network & Internet → Wi-Fi → your hotspot name → Properties
2. Edit IP assignment → **Manual**
3. Set:
   - IP address: `172.20.10.4`
   - Subnet mask: `255.255.255.0`
   - Gateway: `172.20.10.1` (usually the phone)
   - DNS: `8.8.8.8` (or leave automatic)
4. Save, then verify: `ipconfig` should show `172.20.10.4` on the Wi-Fi adapter.

If your phone hotspot uses a different subnet (e.g. `192.168.x.x`), either:

- Change the laptop static IP to match that subnet **and** reflash the MPPT Pico with an updated `SUN_SERVER_HOST`, or
- Use a hotspot that assigns `172.20.10.x` addresses.

## Configure Pico IPs in `hardware/config.py`

Power on each Pico and read USB serial:

```
Pico IP: 172.20.10.5
```

Set:

- `PICO_URL_PV` → MPPT Pico IP (`GET /pout`)
- `PICO_URL_CP` → Capacitor Pico IP (`GET /pout`, `GET /data` for voltage)

Do **not** put the laptop IP in `PICO_URL_PV`.

## Run order

```powershell
# Terminal 1 — required for MPPT Pico
python -m webserver.middleware.app

# Terminal 2 — RL inference
cd "RL Training"
python inference.py
```

## Action commands (laptop → Pico)

`inference.py` sends each action with **GET** and a query parameter:

```
GET http://172.20.10.2/action?value=-2.997
```

On the Pico, handle in `handle_web_request()` (same pattern as `GET /pout`):

```python
elif "GET /action" in request:
    value_str = None
    marker = "value="
    idx = request.find(marker)
    if idx >= 0:
        rest = request[idx + len(marker):]
        value_str = rest.split()[0].split("&")[0].strip()
    if value_str:
        pout_ref = float(value_str)
    body = "OK"
```

Set `ACTIONS_ENABLED = True` in `hardware/config.py` once this handler exists.

**Cap Pico (`sc`):** `value` = `sc_action × 3` (watts).

**Load Pico (`def`):** `value` = total demand power to serve:

```
instant_demand + (total_deferrable_energy_J × def_action / 5)
```

`def_action` is the model fraction in `[0, 1]`; `/5` converts energy-per-tick to power (W).

## Quick connectivity tests (from laptop)

```powershell
curl http://172.20.10.4:8000/api/sun_data
curl http://<MPPT_PICO_IP>/pout
curl http://<CAP_PICO_IP>/pout
curl http://<CAP_PICO_IP>/data
curl "http://<CAP_PICO_IP>/action?value=-1.5"
curl https://icelec50015.azurewebsites.net/price
```
