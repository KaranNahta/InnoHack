# Configuration settings for Agmarknet Ingestion module

DEFAULT_COMMODITIES = [
    "Rice", "Wheat", "Potato", "Onion", "Tomato", "Gram Dal",
    "Mustard Oil", "Sugar", "Turmeric", "Cotton", "Maize",
    "Soyabean", "Groundnut", "Moong Dal", "Urad Dal", "Apple"
]

# Default target states (None or empty list means all states)
DEFAULT_STATES = [
    "Uttar Pradesh", "Punjab", "Haryana", "Maharashtra", "Gujarat",
    "Karnataka", "Madhya Pradesh", "Rajasthan", "Tamil Nadu", "Andhra Pradesh",
    "Bihar", "West Bengal", "Kerala", "Telangana", "Odisha"
]

# Rate-limiting parameters
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_DELAY_SECONDS = 1.0

# API mock/sandbox configurations
MOCK_API_FAIL_PROBABILITY = 0.05  # probability of transient failure to test retries
