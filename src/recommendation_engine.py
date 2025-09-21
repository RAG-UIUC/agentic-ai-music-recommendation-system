"""
Music recommendation engine using collaborative filtering and content-based approaches.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import (
    Track, UserProfile, RecommendationRequest, 
    Recommendation, RecommendationResponse, Genre, MoodPreference
)


class MusicRecommendationEngine:
    """
    AI-powered music recommendation engine that combines collaborative filtering,
    content-based filtering, and user preference analysis.
    """
    
    def __init__(self):
        self.tracks_db: Dict[str, Track] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_matrix: Optional[pd.DataFrame] = None
        self.content_features: Optional[np.ndarray] = None
        self.track_ids: List[str] = []
        
    def add_track(self, track: Track) -> None:
        """Add a track to the music database."""
        self.tracks_db[track.id] = track
        
    def add_user_profile(self, user_profile: UserProfile) -> None:
        """Add or update a user profile."""
        self.user_profiles[user_profile.user_id] = user_profile
        
    def _build_content_features(self) -> None:
        """Build content-based features for tracks."""
        if not self.tracks_db:
            return
            
        track_features = []
        self.track_ids = list(self.tracks_db.keys())
        
        for track_id in self.track_ids:
            track = self.tracks_db[track_id]
            
            # Create feature vector: genre, mood, energy, popularity, year
            genre_vector = [1 if track.genre == genre else 0 for genre in Genre]
            mood_vector = [1 if mood in track.mood_tags else 0 for mood in MoodPreference]
            
            # Normalize year (assuming range 1950-2024)
            normalized_year = (track.release_year - 1950) / (2024 - 1950)
            
            feature_vector = (
                genre_vector + 
                mood_vector + 
                [track.energy_level, track.popularity_score, normalized_year]
            )
            track_features.append(feature_vector)
            
        self.content_features = np.array(track_features)
        
    def _calculate_user_preferences_vector(self, user_profile: UserProfile) -> np.ndarray:
        """Calculate user preference vector based on their profile."""
        if self.content_features is None:
            self._build_content_features()
            
        # Create user preference vector
        genre_prefs = [1 if genre in user_profile.preferred_genres else 0 for genre in Genre]
        mood_prefs = [1 if mood in user_profile.preferred_moods else 0 for mood in MoodPreference]
        
        user_vector = np.array(
            genre_prefs + 
            mood_prefs + 
            [user_profile.energy_preference, 0.5, 0.5]  # neutral popularity and year preference
        )
        
        return user_vector
        
    def _content_based_recommendations(
        self, 
        user_profile: UserProfile, 
        num_recommendations: int
    ) -> List[tuple]:
        """Generate recommendations using content-based filtering."""
        if self.content_features is None:
            self._build_content_features()
            
        user_vector = self._calculate_user_preferences_vector(user_profile)
        
        # Calculate similarity between user preferences and tracks
        similarities = cosine_similarity([user_vector], self.content_features)[0]
        
        # Get track indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            track_id = self.track_ids[idx]
            
            # Skip tracks the user has already heard or disliked
            if (track_id in user_profile.listening_history or 
                track_id in user_profile.disliked_tracks):
                continue
                
            track = self.tracks_db[track_id]
            confidence = similarities[idx]
            
            # Generate explanation
            reason = self._generate_recommendation_reason(track, user_profile, confidence)
            
            recommendations.append((track, confidence, reason))
            
            if len(recommendations) >= num_recommendations:
                break
                
        return recommendations
        
    def _generate_recommendation_reason(
        self, 
        track: Track, 
        user_profile: UserProfile, 
        confidence: float
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        reasons = []
        
        # Genre match
        if track.genre in user_profile.preferred_genres:
            reasons.append(f"matches your preference for {track.genre.value}")
            
        # Mood match
        mood_matches = [mood for mood in track.mood_tags if mood in user_profile.preferred_moods]
        if mood_matches:
            mood_str = ", ".join([mood.value for mood in mood_matches])
            reasons.append(f"fits your {mood_str} mood preference")
            
        # Energy level match
        energy_diff = abs(track.energy_level - user_profile.energy_preference)
        if energy_diff < 0.3:
            reasons.append("matches your energy level preference")
            
        # High popularity
        if track.popularity_score > 0.7:
            reasons.append("is a popular track")
            
        if not reasons:
            reasons.append(f"has a {confidence:.1%} compatibility with your taste")
            
        return f"Recommended because it {' and '.join(reasons)}"
        
    def _context_based_adjustments(
        self, 
        recommendations: List[tuple], 
        context: Optional[str]
    ) -> List[tuple]:
        """Adjust recommendations based on context (workout, study, etc.)."""
        if not context:
            return recommendations
            
        context_preferences = {
            "workout": {"energy_boost": 0.3, "preferred_moods": [MoodPreference.ENERGETIC]},
            "study": {"energy_boost": -0.2, "preferred_moods": [MoodPreference.FOCUSED, MoodPreference.RELAXED]},
            "party": {"energy_boost": 0.4, "preferred_moods": [MoodPreference.ENERGETIC, MoodPreference.HAPPY]},
            "romantic": {"energy_boost": -0.1, "preferred_moods": [MoodPreference.ROMANTIC]},
            "chill": {"energy_boost": -0.3, "preferred_moods": [MoodPreference.RELAXED]}
        }
        
        if context.lower() not in context_preferences:
            return recommendations
            
        prefs = context_preferences[context.lower()]
        adjusted_recommendations = []
        
        for track, confidence, reason in recommendations:
            adjusted_confidence = confidence
            
            # Adjust based on energy level
            if "energy_boost" in prefs:
                energy_match = 1 - abs(track.energy_level - (0.5 + prefs["energy_boost"]))
                adjusted_confidence *= energy_match
                
            # Adjust based on mood preference
            if "preferred_moods" in prefs:
                mood_match = any(mood in track.mood_tags for mood in prefs["preferred_moods"])
                if mood_match:
                    adjusted_confidence *= 1.2
                    
            reason += f" (optimized for {context})"
            adjusted_recommendations.append((track, adjusted_confidence, reason))
            
        # Re-sort by adjusted confidence
        adjusted_recommendations.sort(key=lambda x: x[1], reverse=True)
        return adjusted_recommendations
        
    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Generate personalized music recommendations for a user.
        """
        user_profile = request.user_profile
        
        # Update user profile in our database
        self.add_user_profile(user_profile)
        
        # Get content-based recommendations
        content_recs = self._content_based_recommendations(
            user_profile, 
            request.num_recommendations * 2  # Get more to allow for filtering
        )
        
        # Apply context-based adjustments
        adjusted_recs = self._context_based_adjustments(content_recs, request.context)
        
        # Convert to Recommendation objects
        recommendations = []
        for track, confidence, reason in adjusted_recs[:request.num_recommendations]:
            rec = Recommendation(
                track=track,
                confidence_score=confidence,
                reason=reason
            )
            recommendations.append(rec)
            
        return RecommendationResponse(
            user_id=user_profile.user_id,
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            generated_at=datetime.now().isoformat()
        )
        
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self.user_profiles.get(user_id)
        
    def update_user_interaction(self, user_id: str, track_id: str, liked: bool = True) -> None:
        """Update user interaction with a track (like/dislike)."""
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
            
            if liked:
                if track_id not in user_profile.listening_history:
                    user_profile.listening_history.append(track_id)
                # Remove from disliked if it was there
                if track_id in user_profile.disliked_tracks:
                    user_profile.disliked_tracks.remove(track_id)
            else:
                if track_id not in user_profile.disliked_tracks:
                    user_profile.disliked_tracks.append(track_id)
                # Remove from listening history if it was there
                if track_id in user_profile.listening_history:
                    user_profile.listening_history.remove(track_id)
                    
    def get_track_stats(self) -> Dict:
        """Get statistics about the track database."""
        if not self.tracks_db:
            return {"total_tracks": 0}
            
        genres = [track.genre for track in self.tracks_db.values()]
        genre_counts = {genre.value: genres.count(genre) for genre in Genre}
        
        avg_popularity = np.mean([track.popularity_score for track in self.tracks_db.values()])
        avg_energy = np.mean([track.energy_level for track in self.tracks_db.values()])
        
        return {
            "total_tracks": len(self.tracks_db),
            "genre_distribution": genre_counts,
            "average_popularity": avg_popularity,
            "average_energy": avg_energy,
            "total_users": len(self.user_profiles)
        }