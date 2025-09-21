# Agentic AI Music Recommendation System

An intelligent music recommendation system that uses AI to provide personalized music suggestions based on user preferences, listening history, and contextual factors.

## Features

- **Personalized Recommendations**: Uses content-based filtering to match user preferences
- **Context-Aware**: Adapts recommendations based on context (workout, study, party, etc.)
- **User Profile Management**: Tracks user preferences, listening history, and feedback
- **Multiple Genres & Moods**: Supports various music genres and mood preferences
- **AI-Powered**: Uses machine learning algorithms for similarity matching
- **CLI Interface**: Easy-to-use command-line interface for testing

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RAG-UIUC/agentic-ai-music-recommendation-system.git
cd agentic-ai-music-recommendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Get recommendations for a sample user:
```bash
python cli.py --user user_001
```

#### Get context-specific recommendations:
```bash
python cli.py --user user_001 --context workout
```

#### Create a custom user profile:
```bash
python cli.py --custom
```

#### View system statistics:
```bash
python cli.py --stats
```

### Example Output

```
🎵 Music Recommendations for User: user_001
Generated at: 2024-01-15T10:30:00
Total recommendations: 5
================================================================================

1. Sweet Child O' Mine by Guns N' Roses
   Album: Appetite for Destruction
   Genre: Rock
   Mood: energetic, happy
   Energy Level: 0.9/1.0
   Popularity: 91%
   Confidence: 85%
   Why: Recommended because it matches your preference for rock and fits your energetic, happy mood preference
```

## Architecture

### Core Components

- **`src/models.py`**: Data models for tracks, users, and recommendations
- **`src/recommendation_engine.py`**: Main AI recommendation engine
- **`data/sample_data.py`**: Sample tracks and user profiles for testing
- **`cli.py`**: Command-line interface

### Supported Features

#### Music Genres
- Rock, Pop, Jazz, Classical, Electronic, Hip Hop, Country, Blues, Reggae, Folk

#### Mood Preferences
- Energetic, Relaxed, Happy, Melancholic, Focused, Romantic

#### Context Types
- Workout, Study, Party, Romantic, Chill

## How It Works

1. **Content-Based Filtering**: Analyzes track features (genre, mood, energy, popularity) and user preferences
2. **Similarity Matching**: Uses cosine similarity to find tracks that match user taste
3. **Context Adjustment**: Modifies recommendations based on situational context
4. **Personalization**: Learns from user feedback and listening history

## Testing

Run the test suite:
```bash
python tests/test_recommendation_system.py
```

## API Usage

```python
from src.recommendation_engine import MusicRecommendationEngine
from src.models import UserProfile, RecommendationRequest, Genre, MoodPreference

# Initialize engine
engine = MusicRecommendationEngine()

# Create user profile
user_profile = UserProfile(
    user_id="john_doe",
    preferred_genres=[Genre.ROCK, Genre.POP],
    preferred_moods=[MoodPreference.ENERGETIC, MoodPreference.HAPPY],
    energy_preference=0.8
)

# Get recommendations
request = RecommendationRequest(
    user_profile=user_profile,
    num_recommendations=5,
    context="workout"
)

response = engine.get_recommendations(request)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is private and proprietary to RAG-UIUC.