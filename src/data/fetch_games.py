import requests
import time
import datetime
from pathlib import Path

from config import USERNAME, PGN_FILE

# Delay between API requests (in seconds) to satisfy Chess.com rate limiting guidelines
DELAY_SECONDS = 1.0

# Base URL for Chess.com Public REST API
CHESS_COM_BASE_URL = "https://api.chess.com/pub/player"

# Custom HTTP headers including User-Agent identifier
HTTP_HEADERS = {
    "User-Agent": f"chess-bot-project/2.0 (learning project; user={USERNAME})"
}


def fetch_user_archive_urls(username: str) -> list[str]:
    """
    Fetches the list of monthly archive URLs for a given Chess.com player.

    Args:
      username: Target Chess.com username.

    Returns:
      List of monthly game archive URLs.
    """
    url = f"{CHESS_COM_BASE_URL}/{username}/games/archives"
    response = requests.get(url, headers=HTTP_HEADERS)
    response.raise_for_status()
    archives = response.json().get("archives", [])
    print(f"Found {len(archives)} monthly archives for {username}")
    return archives


def get_archives(username: str) -> list[str]:
    """Backward-compatible wrapper for fetch_user_archive_urls."""
    return fetch_user_archive_urls(username)


def download_and_cache_pgn_games(username: str = USERNAME, pgn_file_path: str = PGN_FILE) -> None:
    """
    Downloads all game archives for the target user from Chess.com, caches monthly PGN files
    locally, and merges them into a single master PGN file.

    Args:
      username: Target Chess.com username.
      pgn_file_path: Output file path for the merged master PGN file.
    """
    print(f"\nFetching games for player: {username}")
    archives = fetch_user_archive_urls(username)

    if not archives:
        print("No archives found. Please verify the username.")
        return

    out_file = Path(pgn_file_path)
    out_dir = out_file.parent
    cache_dir = out_dir / f"cache_{username}"

    # Ensure output and local cache directories exist
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine current month string (current month is re-downloaded to include recent games)
    now = datetime.datetime.now()
    current_month_str = f"{now.year}/{now.month:02d}"

    all_pgn_chunks = []
    total_games = 0

    for i, archive_url in enumerate(archives):
        month_path = "/".join(archive_url.split("/")[-2:])
        month_filename = month_path.replace("/", "_") + ".pgn"
        cache_file = cache_dir / month_filename

        # If cached file exists and it's not the active current month, read from cache
        if cache_file.exists() and month_path != current_month_str:
            print(f"  [{i+1}/{len(archives)}] {month_path}  ->  (Loaded from local cache)")
            pgn_text = cache_file.read_text(encoding="utf-8")
        else:
            try:
                pgn_url = archive_url + "/pgn"
                response = requests.get(pgn_url, headers=HTTP_HEADERS)
                response.raise_for_status()
                pgn_text = response.text

                # Save downloaded PGN text to local cache file
                cache_file.write_text(pgn_text, encoding="utf-8")
                game_count = pgn_text.count("[Event ")
                print(f"  [{i+1}/{len(archives)}] {month_path}  ->  Downloaded {game_count} games")

                # Respectful API sleep delay
                time.sleep(DELAY_SECONDS)
            except requests.HTTPError as e:
                print(f"  [{i+1}/{len(archives)}] {month_path}  ->  HTTP ERROR: {e}")
                continue

        game_count = pgn_text.count("[Event ")
        total_games += game_count
        if game_count > 0:
            all_pgn_chunks.append(pgn_text)

    # Merge all monthly PGN text blocks into single master file
    print(f"\nMerging {len(archives)} months into {pgn_file_path}...")
    merged_pgn_text = "\n\n".join(chunk.strip() for chunk in all_pgn_chunks)

    out_file.write_text(merged_pgn_text, encoding="utf-8")
    file_size_kb = out_file.stat().st_size / 1024

    print(f"Download & Merge Complete:")
    print(f"  Total games fetched : {total_games}")
    print(f"  Master PGN File     : {out_file}")
    print(f"  File size           : {file_size_kb:.1f} KB")


def fetch_user_games(username: str = USERNAME, pgn_file_path: str = PGN_FILE) -> None:
    """Backward-compatible wrapper for download_and_cache_pgn_games."""
    download_and_cache_pgn_games(username, pgn_file_path)


def main() -> None:
    """Main execution point when running script directly."""
    download_and_cache_pgn_games(USERNAME, PGN_FILE)


if __name__ == "__main__":
    main()
