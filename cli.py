#!/usr/bin/env python3
"""
Command-line interface for the music recommendation system.
"""
import sys
import argparse
import json
from typing import Optional

# Add the project root to the Python path
sys.path.append('/home/runner/work/agentic-ai-music-recommendation-system/agentic-ai-music-recommendation-system')

from src.recommendation_engine import MusicRecommendationEngine
from src.models import UserProfile, RecommendationRequest, Genre, MoodPreference
from data.sample_data import get_sample_tracks, get_sample_users


def setup_engine_with_sample_data():
    """Initialize the recommendation engine with sample data."""
    engine = MusicRecommendationEngine()
    
    # Add sample tracks
    for track in get_sample_tracks():
        engine.add_track(track)
    
    # Add sample users
    for user in get_sample_users():
        engine.add_user_profile(user)
    
    return engine


def display_recommendations(response):
    """Display recommendations in a user-friendly format."""
    print(f"\n🎵 Music Recommendations for User: {response.user_id}")
    print(f"Generated at: {response.generated_at}")
    print(f"Total recommendations: {response.total_recommendations}")
    print("=" * 80)
    
    for i, rec in enumerate(response.recommendations, 1):
        track = rec.track
        print(f"\n{i}. {track.title} by {track.artist}")
        print(f"   Album: {track.album}")
        print(f"   Genre: {track.genre.value.title()}")
        print(f"   Mood: {', '.join([mood.value for mood in track.mood_tags])}")
        print(f"   Energy Level: {track.energy_level:.1f}/1.0")
        print(f"   Popularity: {track.popularity_score:.1%}")
        print(f"   Confidence: {rec.confidence_score:.1%}")
        print(f"   Why: {rec.reason}")


def create_custom_user():
    """Interactive function to create a custom user profile."""
    print("\n🎯 Create Your Music Profile")
    print("=" * 40)
    
    user_id = input("Enter your user ID: ").strip()
    
    # Genre preferences
    print("\nAvailable genres:")
    genres = list(Genre)
    for i, genre in enumerate(genres, 1):
        print(f"  {i}. {genre.value.title()}")
    
    genre_choices = input("\nEnter genre numbers (comma-separated, e.g., 1,3,5): ").strip()
    preferred_genres = []
    for choice in genre_choices.split(','):
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(genres):
                preferred_genres.append(genres[idx])
        except ValueError:
            continue
    
    # Mood preferences
    print("\nAvailable moods:")
    moods = list(MoodPreference)
    for i, mood in enumerate(moods, 1):
        print(f"  {i}. {mood.value.title()}")
    
    mood_choices = input("\nEnter mood numbers (comma-separated, e.g., 1,2,4): ").strip()
    preferred_moods = []
    for choice in mood_choices.split(','):
        try:
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(moods):
                preferred_moods.append(moods[idx])
        except ValueError:
            continue
    
    # Energy preference
    energy_input = input("\nEnergy preference (0.0 for low energy, 1.0 for high energy, default 0.5): ").strip()
    try:
        energy_preference = float(energy_input) if energy_input else 0.5
        energy_preference = max(0.0, min(1.0, energy_preference))
    except ValueError:
        energy_preference = 0.5
    
    age_range = input("Age range (optional, e.g., '25-35'): ").strip() or None
    
    return UserProfile(
        user_id=user_id,
        preferred_genres=preferred_genres,
        preferred_moods=preferred_moods,
        age_range=age_range,
        energy_preference=energy_preference
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Music Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --user user_001                    # Get recommendations for sample user
  %(prog)s --user user_002 --context workout # Get workout recommendations
  %(prog)s --custom                           # Create custom user profile
  %(prog)s --stats                            # Show system statistics
        """
    )
    
    parser.add_argument('--user', type=str, help='User ID for recommendations')
    parser.add_argument('--custom', action='store_true', help='Create custom user profile')
    parser.add_argument('--context', type=str, choices=['workout', 'study', 'party', 'romantic', 'chill'],
                       help='Context for recommendations')
    parser.add_argument('--num', type=int, default=5, help='Number of recommendations (default: 5)')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    
    args = parser.parse_args()
    
    # Initialize the recommendation engine
    print("🎼 Initializing AI Music Recommendation System...")
    engine = setup_engine_with_sample_data()
    
    if args.stats:
        print("\n📊 System Statistics")
        print("=" * 30)
        stats = engine.get_track_stats()
        print(f"Total tracks in database: {stats['total_tracks']}")
        print(f"Total registered users: {stats['total_users']}")
        print(f"Average track popularity: {stats['average_popularity']:.1%}")
        print(f"Average energy level: {stats['average_energy']:.1f}/1.0")
        print("\nGenre distribution:")
        for genre, count in stats['genre_distribution'].items():
            print(f"  {genre.title()}: {count} tracks")
        return
    
    # Get user profile
    user_profile = None
    
    if args.custom:
        user_profile = create_custom_user()
    elif args.user:
        user_profile = engine.get_user_profile(args.user)
        if not user_profile:
            print(f"❌ User '{args.user}' not found.")
            print("Available sample users: user_001, user_002, user_003")
            print("Or use --custom to create a new profile.")
            return
    else:
        print("❌ Please specify --user <user_id> or --custom to create a profile.")
        print("Use --help for more options.")
        return
    
    # Generate recommendations
    request = RecommendationRequest(
        user_profile=user_profile,
        num_recommendations=args.num,
        context=args.context
    )
    
    print(f"\n🤖 Generating recommendations...")
    if args.context:
        print(f"Context: {args.context}")
    
    response = engine.get_recommendations(request)
    display_recommendations(response)
    
    # Show user profile summary
    print(f"\n👤 User Profile Summary:")
    print(f"   Preferred genres: {', '.join([g.value for g in user_profile.preferred_genres])}")
    print(f"   Preferred moods: {', '.join([m.value for m in user_profile.preferred_moods])}")
    print(f"   Energy preference: {user_profile.energy_preference:.1f}/1.0")
    if user_profile.age_range:
        print(f"   Age range: {user_profile.age_range}")


if __name__ == "__main__":
    main()