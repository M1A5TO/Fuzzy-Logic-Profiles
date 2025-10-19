# Testy klasyfikacji stacji kolej/metro z warunkiem 2-tagowym
def is_train_station(tags):
    railway = tags.get("railway")
    pub = tags.get("public_transport")
    return (railway in {"station", "halt"}) and (pub in {"station", "halt"})

def classify_row(row):
    if is_train_station({"railway": row.get("railway"), "public_transport": row.get("public_transport")}):
        return "stacja_kolej_metro"
    if row.get("railway") == "tram_stop":
        return "przystanek_tramwaj"
    if row.get("highway") == "bus_stop":
        return "przystanek_autobus"
    return None

def test_train_station_requires_both_tags():
    assert classify_row({"railway":"station", "public_transport":"station"}) == "stacja_kolej_metro"
    assert classify_row({"railway":"halt",    "public_transport":"station"}) == "stacja_kolej_metro"
    assert classify_row({"railway":"station", "public_transport":"halt"})    == "stacja_kolej_metro"

def test_train_station_rejects_single_tag():
    assert classify_row({"railway":"station", "public_transport":None}) is None
    assert classify_row({"railway":None, "public_transport":"station"}) is None

def test_tram_and_bus_stops():
    assert classify_row({"railway":"tram_stop"}) == "przystanek_tramwaj"
    assert classify_row({"highway":"bus_stop"})  == "przystanek_autobus"
