# Configuration settings for Agmarknet Ingestion module

# Default commodities to query if none are specified
DEFAULT_COMMODITIES = [
    "Rice",
    "Wheat",
    "Potato",
    "Onion",
    "Tomato",
    "Pulses"
]

# Default target states (None or empty list means all states)
DEFAULT_STATES = [
    "Uttar Pradesh",
    "Punjab",
    "Haryana",
    "Maharashtra",
    "Gujarat"
]

# Rate-limiting parameters
RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_DELAY_SECONDS = 1.0

# API mock/sandbox configurations
MOCK_API_FAIL_PROBABILITY = 0.05  # probability of transient failure to test retries
