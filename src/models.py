"""
Data models for the music recommendation system.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum


class Genre(str, Enum):
    """Music genres supported by the system."""
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    HIP_HOP = "hip_hop"
    COUNTRY = "country"
    BLUES = "blues"
    REGGAE = "reggae"
    FOLK = "folk"


class MoodPreference(str, Enum):
    """User mood preferences."""
    ENERGETIC = "energetic"
    RELAXED = "relaxed"
    HAPPY = "happy"
    MELANCHOLIC = "melancholic"
    FOCUSED = "focused"
    ROMANTIC = "romantic"


class Track(BaseModel):
    """Represents a music track."""
    id: str
    title: str
    artist: str
    album: str
    genre: Genre
    duration: int  # in seconds
    mood_tags: List[MoodPreference]
    popularity_score: float  # 0.0 to 1.0
    energy_level: float  # 0.0 to 1.0
    release_year: int


class UserProfile(BaseModel):
    """Represents a user's profile and preferences."""
    user_id: str
    preferred_genres: List[Genre]
    preferred_moods: List[MoodPreference]
    age_range: Optional[str] = None
    listening_history: List[str] = []  # Track IDs
    disliked_tracks: List[str] = []  # Track IDs
    energy_preference: float = 0.5  # 0.0 (low energy) to 1.0 (high energy)


class RecommendationRequest(BaseModel):
    """Request for music recommendations."""
    user_profile: UserProfile
    num_recommendations: int = 10
    context: Optional[str] = None  # e.g., "workout", "study", "party"
    exclude_recent: bool = True


class Recommendation(BaseModel):
    """A single music recommendation."""
    track: Track
    confidence_score: float  # 0.0 to 1.0
    reason: str  # Explanation for the recommendation


class RecommendationResponse(BaseModel):
    """Response containing music recommendations."""
    user_id: str
    recommendations: List[Recommendation]
    total_recommendations: int
    generated_at: str