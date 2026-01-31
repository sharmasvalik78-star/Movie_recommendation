import requests

API_KEY = "2b6915d50fc8b583807848395532c416"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def safe_get(url, params=None):
    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

import re

def search_movie(title):
    # Extract year like (2007)
    year_match = re.search(r"\((\d{4})\)", title)
    year = year_match.group(1) if year_match else None

    # Remove year from title
    clean_title = re.sub(r"\(\d{4}\)", "", title).strip()

    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": clean_title,
        "year": year
    }

    data = safe_get(url, params)

    if not data or not data.get("results"):
        return None

    # Choose best match by popularity
    return sorted(
        data["results"],
        key=lambda x: x.get("popularity", 0),
        reverse=True
    )[0]

def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY}
    return safe_get(url, params)

def get_movie_trailer(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/videos"
    params = {"api_key": API_KEY}
    data = safe_get(url, params)

    if not data:
        return None

    for video in data.get("results", []):
        if video["type"] == "Trailer" and video["site"] == "YouTube":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

def get_poster_path(poster_path):
    return IMAGE_BASE + poster_path if poster_path else None

def get_movie_by_tmdb_id(tmdb_id):
    if tmdb_id is None:
        return None

    url = f"{BASE_URL}/movie/{int(tmdb_id)}"
    params = {"api_key": API_KEY}
    return safe_get(url, params)

