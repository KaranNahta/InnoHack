import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.config import RATE_LIMIT_DELAY_SECONDS, MOCK_API_FAIL_PROBABILITY

class AgmarknetClient:
    def __init__(self, api_key: str = None, base_url: str = None, simulate_failures: bool = True):
        self.api_key = api_key
        self.base_url = base_url
        self.simulate_failures = simulate_failures

    def fetch_raw_data(
        self,
        start_date: str,
        end_date: str,
        commodities: List[str] = None,
        states: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches raw mandi data from Agmarknet API for the given dates, commodities, and states.
        In sandbox/mock mode, it generates realistic mock data with some flaws (zeros, missing values).
        """
        # Simulate rate-limiting delay
        time.sleep(RATE_LIMIT_DELAY_SECONDS)

        # Simulate transient failure to test exponential backoff
        if self.simulate_failures and random.random() < MOCK_API_FAIL_PROBABILITY:
            raise RuntimeError("Transient API Error: 503 Service Unavailable")

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Define default options if none provided
        if not commodities:
            commodities = ["Rice", "Wheat", "Potato", "Onion", "Tomato", "Pulses"]
        if not states:
            states = ["Uttar Pradesh", "Punjab", "Haryana", "Maharashtra", "Gujarat"]

        # Map states to some districts and markets
        state_geo = {
            "Uttar Pradesh": [("Agra", "Agra Mandi"), ("Kanpur", "Kanpur Mandi"), ("Lucknow", "Lucknow Mandi")],
            "Punjab": [("Amritsar", "Amritsar Mandi"), ("Ludhiana", "Ludhiana Mandi"), ("Patiala", "Patiala Mandi")],
            "Haryana": [("Karnal", "Karnal Mandi"), ("Ambala", "Ambala Mandi"), ("Rohtak", "Rohtak Mandi")],
            "Maharashtra": [("Pune", "Pune Mandi"), ("Mumbai", "Kalyan Mandi"), ("Nashik", "Nashik Mandi")],
            "Gujarat": [("Ahmedabad", "Ahmedabad Mandi"), ("Surat", "Surat Mandi"), ("Rajkot", "Rajkot Mandi")]
        }

        # Map commodities to varieties
        commodity_varieties = {
            "Rice": ["Basmati", "Common", "Permal"],
            "Wheat": ["Kalyan", "Desi", "Lok-1"],
            "Potato": ["Jyoti", "Local", "Desi"],
            "Onion": ["Red", "White", "Nasik"],
            "Tomato": ["Local", "Hybrid"],
            "Pulses": ["Arhar (Tur)", "Masur", "Urad"]
        }

        records = []
        delta = end - start
        
        # Seed generator for reproducibility in testing (if we want, but simple random is fine. We can override in tests)
        for i in range(delta.days + 1):
            current_date = start + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            for state in states:
                # Get districts/markets for this state
                geo_list = state_geo.get(state, [("Default District", "Default Mandi")])
                for district, market in geo_list:
                    for commodity in commodities:
                        # Mandi usually closed on Sundays
                        if current_date.weekday() == 6 and random.random() < 0.9:
                            continue
                        if random.random() < 0.15:
                            continue  # Random missing report
                        
                        varieties = commodity_varieties.get(commodity, ["Other"])
                        for variety in varieties:
                            # Generate base prices per quintal
                            base_price = {
                                "Rice": 3000,
                                "Wheat": 2200,
                                "Potato": 1200,
                                "Onion": 1800,
                                "Tomato": 1500,
                                "Pulses": 6500
                            }.get(commodity, 2000)

                            min_price = base_price + random.randint(-200, 200)
                            max_price = min_price + random.randint(100, 500)
                            modal_price = (min_price + max_price) / 2.0

                            # Arrivals in quintals (we will convert it to tonnes in the engine)
                            arrivals = float(round(random.uniform(10, 500), 2))

                            # Introduce data issues
                            issue_roll = random.random()
                            if issue_roll < 0.05:
                                modal_price = 0.0
                            elif issue_roll < 0.05:
                                modal_price = None

                            records.append({
                                "state_name": state,
                                "district_name": district,
                                "market_name": market,
                                "commodity": commodity,
                                "variety": variety,
                                "min_price": float(min_price),
                                "max_price": float(max_price),
                                "modal_price": float(modal_price) if modal_price is not None else None,
                                "arrivals": arrivals,
                                "reported_date": date_str
                            })

        return records
