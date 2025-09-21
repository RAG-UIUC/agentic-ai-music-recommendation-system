# Configuration file for the music recommendation system

# Database settings
DATABASE_URL = "sqlite:///music_recommendations.db"

# AI/ML settings
DEFAULT_SIMILARITY_THRESHOLD = 0.5
MAX_RECOMMENDATIONS = 50
DEFAULT_RECOMMENDATIONS = 10

# Content-based filtering weights
GENRE_WEIGHT = 0.3
MOOD_WEIGHT = 0.25
ENERGY_WEIGHT = 0.2
POPULARITY_WEIGHT = 0.15
YEAR_WEIGHT = 0.1

# Context adjustments
CONTEXT_ENERGY_ADJUSTMENTS = {
    "workout": 0.3,
    "study": -0.2,
    "party": 0.4,
    "romantic": -0.1,
    "chill": -0.3
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
ENABLE_CORS = True

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "music_recommendations.log"