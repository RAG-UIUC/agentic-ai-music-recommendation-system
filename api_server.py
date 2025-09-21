"""
FastAPI server for the music recommendation system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.append('/home/runner/work/agentic-ai-music-recommendation-system/agentic-ai-music-recommendation-system')

from src.recommendation_engine import MusicRecommendationEngine
from src.models import (
    UserProfile, RecommendationRequest, RecommendationResponse,
    Track, Genre, MoodPreference
)
from data.sample_data import get_sample_tracks, get_sample_users

app = FastAPI(
    title="AI Music Recommendation System",
    description="An intelligent music recommendation system using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the recommendation engine
engine = MusicRecommendationEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize the system with sample data."""
    # Load sample tracks
    for track in get_sample_tracks():
        engine.add_track(track)
    
    # Load sample users
    for user in get_sample_users():
        engine.add_user_profile(user)


@app.get("/")
async def root():
    """Root endpoint with system information."""
    stats = engine.get_track_stats()
    return {
        "message": "AI Music Recommendation System API",
        "version": "1.0.0",
        "stats": stats
    }


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized music recommendations."""
    try:
        response = engine.get_recommendations(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile by ID."""
    user_profile = engine.get_user_profile(user_id)
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found")
    return user_profile


@app.post("/users")
async def create_user_profile(user_profile: UserProfile):
    """Create or update a user profile."""
    engine.add_user_profile(user_profile)
    return {"message": "User profile created successfully", "user_id": user_profile.user_id}


@app.get("/tracks/{track_id}")
async def get_track(track_id: str):
    """Get track by ID."""
    track = engine.tracks_db.get(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@app.get("/tracks")
async def list_tracks():
    """List all tracks in the database."""
    return list(engine.tracks_db.values())


@app.post("/interactions/{user_id}/{track_id}")
async def update_user_interaction(user_id: str, track_id: str, liked: bool = True):
    """Update user interaction with a track."""
    if user_id not in engine.user_profiles:
        raise HTTPException(status_code=404, detail="User not found")
    
    if track_id not in engine.tracks_db:
        raise HTTPException(status_code=404, detail="Track not found")
    
    engine.update_user_interaction(user_id, track_id, liked)
    return {"message": "Interaction updated successfully"}


@app.get("/stats")
async def get_system_stats():
    """Get system statistics."""
    return engine.get_track_stats()


@app.get("/genres")
async def list_genres():
    """List all supported genres."""
    return [genre.value for genre in Genre]


@app.get("/moods")
async def list_moods():
    """List all supported mood preferences."""
    return [mood.value for mood in MoodPreference]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)