"""
Basic tests for the music recommendation system.
"""
import sys
sys.path.append('/home/runner/work/agentic-ai-music-recommendation-system/agentic-ai-music-recommendation-system')

from src.recommendation_engine import MusicRecommendationEngine
from src.models import UserProfile, RecommendationRequest, Genre, MoodPreference
from data.sample_data import get_sample_tracks, get_sample_users


def test_engine_initialization():
    """Test that the engine initializes correctly."""
    engine = MusicRecommendationEngine()
    assert len(engine.tracks_db) == 0
    assert len(engine.user_profiles) == 0
    print("✅ Engine initialization test passed")


def test_add_tracks_and_users():
    """Test adding tracks and users to the engine."""
    engine = MusicRecommendationEngine()
    
    # Add sample data
    tracks = get_sample_tracks()
    users = get_sample_users()
    
    for track in tracks:
        engine.add_track(track)
    
    for user in users:
        engine.add_user_profile(user)
    
    assert len(engine.tracks_db) == len(tracks)
    assert len(engine.user_profiles) == len(users)
    print("✅ Add tracks and users test passed")


def test_recommendations():
    """Test generating recommendations."""
    engine = MusicRecommendationEngine()
    
    # Setup with sample data
    for track in get_sample_tracks():
        engine.add_track(track)
    
    for user in get_sample_users():
        engine.add_user_profile(user)
    
    # Test recommendation for first user
    user_profile = engine.get_user_profile("user_001")
    assert user_profile is not None
    
    request = RecommendationRequest(
        user_profile=user_profile,
        num_recommendations=3
    )
    
    response = engine.get_recommendations(request)
    
    assert response.user_id == "user_001"
    assert len(response.recommendations) <= 3
    assert response.total_recommendations == len(response.recommendations)
    
    # Check that recommendations have required fields
    for rec in response.recommendations:
        assert rec.track is not None
        assert 0 <= rec.confidence_score <= 1
        assert len(rec.reason) > 0
    
    print("✅ Recommendations generation test passed")


def test_context_recommendations():
    """Test context-based recommendations."""
    engine = MusicRecommendationEngine()
    
    # Setup with sample data
    for track in get_sample_tracks():
        engine.add_track(track)
    
    user_profile = UserProfile(
        user_id="test_user",
        preferred_genres=[Genre.ROCK, Genre.ELECTRONIC],
        preferred_moods=[MoodPreference.ENERGETIC],
        energy_preference=0.8
    )
    
    # Test workout context
    request = RecommendationRequest(
        user_profile=user_profile,
        num_recommendations=3,
        context="workout"
    )
    
    response = engine.get_recommendations(request)
    
    assert len(response.recommendations) > 0
    # Workout recommendations should have higher energy
    for rec in response.recommendations:
        assert rec.track.energy_level >= 0.5  # Should be somewhat energetic
        assert "workout" in rec.reason.lower()
    
    print("✅ Context-based recommendations test passed")


def test_user_interaction_updates():
    """Test updating user interactions."""
    engine = MusicRecommendationEngine()
    
    user_profile = UserProfile(
        user_id="test_user",
        preferred_genres=[Genre.POP],
        preferred_moods=[MoodPreference.HAPPY],
        energy_preference=0.5
    )
    
    engine.add_user_profile(user_profile)
    
    # Test liking a track
    engine.update_user_interaction("test_user", "track_001", liked=True)
    updated_profile = engine.get_user_profile("test_user")
    assert "track_001" in updated_profile.listening_history
    
    # Test disliking a track
    engine.update_user_interaction("test_user", "track_002", liked=False)
    updated_profile = engine.get_user_profile("test_user")
    assert "track_002" in updated_profile.disliked_tracks
    
    print("✅ User interaction updates test passed")


def test_system_stats():
    """Test system statistics."""
    engine = MusicRecommendationEngine()
    
    # Add sample data
    for track in get_sample_tracks():
        engine.add_track(track)
    
    for user in get_sample_users():
        engine.add_user_profile(user)
    
    stats = engine.get_track_stats()
    
    assert stats["total_tracks"] == len(get_sample_tracks())
    assert stats["total_users"] == len(get_sample_users())
    assert "genre_distribution" in stats
    assert "average_popularity" in stats
    assert "average_energy" in stats
    
    print("✅ System statistics test passed")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Music Recommendation System Tests")
    print("=" * 50)
    
    try:
        test_engine_initialization()
        test_add_tracks_and_users()
        test_recommendations()
        test_context_recommendations()
        test_user_interaction_updates()
        test_system_stats()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)