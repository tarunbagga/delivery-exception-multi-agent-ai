# Extracted from the project notebook. Execute after notebook setup.
# Requires: tool, csv, sqlite3, DELIVERY_LOGS_PATH, CUSTOMERS_DB_PATH,
# and playbook_retriever from the setup cells. This is not standalone.

@tool
def read_delivery_logs() -> list[dict]:
    """Read all delivery log rows from CSV. Used by preprocessor only."""
    with open(DELIVERY_LOGS_PATH, "r") as f:
        return list(csv.DictReader(f))

db_conn = sqlite3.connect(CUSTOMERS_DB_PATH)
db_conn.row_factory = sqlite3.Row

@tool
def lookup_customer_profile(customer_id: str, include_pii: bool = False) -> dict:
    """Fetch customer profile from SQLite. Name included only when requested."""
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    if row is None:
        return {}
    profile = dict(row)
    if not include_pii:
        profile.pop("name", None)
    return profile

@tool
def check_locker_availability(zip_code: str, package_size: str) -> list[dict]:
    """Find compatible lockers in the same zip code, with eligibility reasons."""
    size_hierarchy = {"SMALL": 1, "MEDIUM": 2, "LARGE": 3}
    pkg_level = size_hierarchy.get(package_size, 0)
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM lockers WHERE zip_code = ?", (zip_code,))
    rows = cursor.fetchall()
    results = []
    for row in rows:
        locker = dict(row)
        locker_max = size_hierarchy.get(locker["max_package_size"], 0)
        if locker_max < pkg_level:
            locker["eligible"] = False
            locker["reason"] = f"Locker max {locker['max_package_size']} < package {package_size}"
        elif locker["capacity_status"] == "FULL":
            locker["eligible"] = False
            locker["reason"] = "Locker is FULL"
        elif locker["capacity_status"] == "LIMITED" and package_size != "SMALL":
            locker["eligible"] = False
            locker["reason"] = "Locker is LIMITED - only SMALL packages accepted"
        else:
            locker["eligible"] = True
            locker["reason"] = "Compatible"
        results.append(locker)
    return results

@tool
def search_playbook(query: str) -> list[dict]:
    """Retrieve relevant playbook sections with page metadata."""
    docs = playbook_retriever.invoke(query)
    return [
        {"content": d.page_content, "page": d.metadata.get("page", "?"),
         "section": d.metadata.get("section", "")}
        for d in docs
    ]

@tool
def check_escalation_rules(customer_tier: str, exceptions_last_90d: int,
                           attempt_number: int, package_type: str,
                           status_code: str, status_description: str) -> dict:
    """Deterministic escalation rule engine from the POC."""
    import re
    triggers = []
    if attempt_number >= 3:
        triggers.append("AUTOMATIC: 3rd failed delivery attempt")
    if customer_tier == "VIP" and exceptions_last_90d >= 3:
        triggers.append(f"AUTOMATIC: VIP customer with {exceptions_last_90d} exceptions in 90d (>=3)")
    if status_code == "DAMAGED" and package_type == "PERISHABLE":
        triggers.append("AUTOMATIC: Damaged perishable package")
    if status_code == "WEATHER_DELAY" and package_type == "PERISHABLE":
        hour_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:hr|hour|hours)', status_description.lower())
        if hour_matches:
            hours = float(hour_matches[0])
            if hours > 4:
                triggers.append(f"AUTOMATIC: Perishable with {hours}hr delay (>4hr threshold)")
    fraud_keywords = ["vacant", "demolished", "construction site", "empty lot"]
    if status_code == "ADDRESS_ISSUE" and any(kw in status_description.lower() for kw in fraud_keywords):
        triggers.append("AUTOMATIC: Potential fraud - address is vacant/demolished")
    if status_code == "ATTEMPTED" and attempt_number == 2 and exceptions_last_90d > 3:
        triggers.append(f"AUTOMATIC: 2nd failed attempt with {exceptions_last_90d} exceptions in 90d (>3)")
    safety_keywords = ["unsafe", "threat", "aggressive dog", "dog attack", "assault", "weapon", "safety concern"]
    if any(kw in status_description.lower() for kw in safety_keywords):
        triggers.append("AUTOMATIC: Driver reported a safety concern")
    if customer_tier == "STANDARD" and exceptions_last_90d > 5:
        triggers.append(f"DISCRETIONARY: Standard customer with {exceptions_last_90d} exceptions in 90d (>5)")
    if customer_tier == "PREMIUM" and package_type == "PERISHABLE" and status_code == "WEATHER_DELAY":
        triggers.append("DISCRETIONARY: Premium customer with perishable in weather delay")
    return {"has_triggers": len(triggers) > 0,
            "trigger_count": len(triggers), "triggers": triggers}
