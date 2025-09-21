"""
Sample data for testing the music recommendation system.
"""
from src.models import Track, UserProfile, Genre, MoodPreference


def get_sample_tracks():
    """Get a collection of sample tracks for testing."""
    return [
        Track(
            id="track_001",
            title="Bohemian Rhapsody",
            artist="Queen",
            album="A Night at the Opera",
            genre=Genre.ROCK,
            duration=355,
            mood_tags=[MoodPreference.ENERGETIC, MoodPreference.HAPPY],
            popularity_score=0.95,
            energy_level=0.8,
            release_year=1975
        ),
        Track(
            id="track_002",
            title="Shape of You",
            artist="Ed Sheeran",
            album="÷ (Divide)",
            genre=Genre.POP,
            duration=233,
            mood_tags=[MoodPreference.HAPPY, MoodPreference.ROMANTIC],
            popularity_score=0.92,
            energy_level=0.7,
            release_year=2017
        ),
        Track(
            id="track_003",
            title="Take Five",
            artist="Dave Brubeck",
            album="Time Out",
            genre=Genre.JAZZ,
            duration=324,
            mood_tags=[MoodPreference.RELAXED, MoodPreference.FOCUSED],
            popularity_score=0.75,
            energy_level=0.4,
            release_year=1959
        ),
        Track(
            id="track_004",
            title="Clair de Lune",
            artist="Claude Debussy",
            album="Suite Bergamasque",
            genre=Genre.CLASSICAL,
            duration=300,
            mood_tags=[MoodPreference.RELAXED, MoodPreference.ROMANTIC],
            popularity_score=0.82,
            energy_level=0.2,
            release_year=1905
        ),
        Track(
            id="track_005",
            title="Strobe",
            artist="Deadmau5",
            album="For Lack of a Better Name",
            genre=Genre.ELECTRONIC,
            duration=645,
            mood_tags=[MoodPreference.ENERGETIC, MoodPreference.FOCUSED],
            popularity_score=0.78,
            energy_level=0.9,
            release_year=2009
        ),
        Track(
            id="track_006",
            title="Lose Yourself",
            artist="Eminem",
            album="8 Mile Soundtrack",
            genre=Genre.HIP_HOP,
            duration=326,
            mood_tags=[MoodPreference.ENERGETIC, MoodPreference.FOCUSED],
            popularity_score=0.88,
            energy_level=0.85,
            release_year=2002
        ),
        Track(
            id="track_007",
            title="The Sound of Silence",
            artist="Simon & Garfunkel",
            album="Sounds of Silence",
            genre=Genre.FOLK,
            duration=204,
            mood_tags=[MoodPreference.MELANCHOLIC, MoodPreference.RELAXED],
            popularity_score=0.85,
            energy_level=0.3,
            release_year=1965
        ),
        Track(
            id="track_008",
            title="Sweet Child O' Mine",
            artist="Guns N' Roses",
            album="Appetite for Destruction",
            genre=Genre.ROCK,
            duration=356,
            mood_tags=[MoodPreference.ENERGETIC, MoodPreference.HAPPY],
            popularity_score=0.91,
            energy_level=0.9,
            release_year=1987
        ),
        Track(
            id="track_009",
            title="Hotel California",
            artist="Eagles",
            album="Hotel California",
            genre=Genre.ROCK,
            duration=391,
            mood_tags=[MoodPreference.MELANCHOLIC, MoodPreference.RELAXED],
            popularity_score=0.93,
            energy_level=0.6,
            release_year=1976
        ),
        Track(
            id="track_010",
            title="Blinding Lights",
            artist="The Weeknd",
            album="After Hours",
            genre=Genre.POP,
            duration=200,
            mood_tags=[MoodPreference.ENERGETIC, MoodPreference.HAPPY],
            popularity_score=0.96,
            energy_level=0.8,
            release_year=2019
        )
    ]


def get_sample_users():
    """Get sample user profiles for testing."""
    return [
        UserProfile(
            user_id="user_001",
            preferred_genres=[Genre.ROCK, Genre.POP],
            preferred_moods=[MoodPreference.ENERGETIC, MoodPreference.HAPPY],
            age_range="25-35",
            listening_history=["track_001", "track_008"],
            disliked_tracks=[],
            energy_preference=0.8
        ),
        UserProfile(
            user_id="user_002",
            preferred_genres=[Genre.JAZZ, Genre.CLASSICAL],
            preferred_moods=[MoodPreference.RELAXED, MoodPreference.FOCUSED],
            age_range="35-50",
            listening_history=["track_003", "track_004"],
            disliked_tracks=["track_005"],
            energy_preference=0.3
        ),
        UserProfile(
            user_id="user_003",
            preferred_genres=[Genre.ELECTRONIC, Genre.HIP_HOP],
            preferred_moods=[MoodPreference.ENERGETIC, MoodPreference.FOCUSED],
            age_range="18-25",
            listening_history=["track_005", "track_006"],
            disliked_tracks=["track_007"],
            energy_preference=0.9
        )
    ]