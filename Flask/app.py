from datetime import date
import json
import os
import sys
import logging
from flask import Flask, render_template, jsonify
from functools import lru_cache
import subprocess, requests, re, time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variable
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', 'a0f0cd0b5cmshfef96ed37a9cda6p1f67bajsnfcdd16f37df8')

@lru_cache()
def fetch_fanduel(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="fanduel")

@lru_cache()
def fetch_draftkings(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="draftkings")

@lru_cache()
def fetch_betmgm(ttl_hash=None):
    del ttl_hash
    return fetch_game_data(sportsbook="betmgm")

def fetch_game_data(sportsbook="fanduel"):
    """Fetch game data with improved error handling and fixed subprocess call."""
    try:
        # Use python3 and sys.executable for better compatibility
        python_cmd = sys.executable if sys.executable else "python3"
        cmd = [python_cmd, "main.py", "-xgb", f"-odds={sportsbook}"]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd="../", capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            logger.error(f"Subprocess failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return {}
            
        stdout = result.stdout
        
        # Fixed regex patterns - removed + from character class
        data_re = re.compile(r'\n(?P<home_team>[\w ]+)(\((?P<home_confidence>[\d\.]+)%\))? vs (?P<away_team>[\w ]+)(\((?P<away_confidence>[\d\.]+)%\))?: (?P<ou_pick>OVER|UNDER) (?P<ou_value>[\d\.]+) (\((?P<ou_confidence>[\d\.]+)%\))?', re.MULTILINE)
        ev_re = re.compile(r'(?P<team>[\w ]+) EV: (?P<ev>[-\d\.]+)', re.MULTILINE)
        odds_re = re.compile(r'(?P<away_team>[\w ]+) \((?P<away_team_odds>-?\d+)\) @ (?P<home_team>[\w ]+) \((?P<home_team_odds>-?\d+)\)', re.MULTILINE)
        
        games = {}
        for match in data_re.finditer(stdout):
            game_dict = {
                'away_team': match.group('away_team').strip(),
                'home_team': match.group('home_team').strip(),
                'away_confidence': match.group('away_confidence'),
                'home_confidence': match.group('home_confidence'),
                'ou_pick': match.group('ou_pick'),
                'ou_value': match.group('ou_value'),
                'ou_confidence': match.group('ou_confidence')
            }
            
            # Add EV data
            for ev_match in ev_re.finditer(stdout):
                if ev_match.group('team') == game_dict['away_team']:
                    game_dict['away_team_ev'] = ev_match.group('ev')
                if ev_match.group('team') == game_dict['home_team']:
                    game_dict['home_team_ev'] = ev_match.group('ev')
                    
            # Add odds data
            for odds_match in odds_re.finditer(stdout):
                if odds_match.group('away_team') == game_dict['away_team']:
                    game_dict['away_team_odds'] = odds_match.group('away_team_odds')
                if odds_match.group('home_team') == game_dict['home_team']:
                    game_dict['home_team_odds'] = odds_match.group('home_team_odds')

            logger.info(f"Processed game: {game_dict['away_team']} @ {game_dict['home_team']}")
            games[f"{game_dict['away_team']}:{game_dict['home_team']}"] = game_dict
            
        return games
        
    except subprocess.TimeoutExpired:
        logger.error("Subprocess timed out")
        return {}
    except Exception as e:
        logger.error(f"Error fetching game data: {str(e)}")
        return {}


def get_ttl_hash(seconds=600):
    """Return the same value within `seconds` time period"""
    return round(time.time() / seconds)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')


@app.route("/")
def index():
    """Main page with improved error handling."""
    try:
        fanduel = fetch_fanduel(ttl_hash=get_ttl_hash())
        draftkings = fetch_draftkings(ttl_hash=get_ttl_hash())
        betmgm = fetch_betmgm(ttl_hash=get_ttl_hash())

        # Ensure data is not None
        data = {
            "fanduel": fanduel or {},
            "draftkings": draftkings or {},
            "betmgm": betmgm or {}
        }

        return render_template('index.html', today=date.today(), data=data)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        # Return empty data on error
        return render_template('index.html', today=date.today(), data={"fanduel": {}, "draftkings": {}, "betmgm": {}})


def get_player_data(team_abv):
    """Fetch player data for a given team abbreviation with improved error handling."""
    if not team_abv:
        return {'success': False, 'error': 'Team abbreviation is required'}
        
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    querystring = {"teamAbv": team_abv}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName', 'Unknown'),
                    'shortName': player.get('shortName', 'Unknown'),
                    'headshot': player.get('nbaComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos', 'N/A'),
                    'height': player.get('height', 'N/A'),
                    'weight': player.get('weight', 'N/A'),
                    'college': player.get('college', 'N/A'),
                    'experience': player.get('exp', 'N/A'),
                    'jerseyNum': player.get('jerseyNum', 'N/A'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay', 'N/A')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': f'API returned status code: {data.get("statusCode", "unknown")}'
        }
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout in get_player_data")
        return {'success': False, 'error': 'Request timeout'}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in get_player_data: {str(e)}")
        return {'success': False, 'error': 'Network error'}
    except Exception as e:
        logger.error(f"Error in get_player_data: {str(e)}")
        return {'success': False, 'error': str(e)}


@app.route("/team-data/<team_name>")
def team_data(team_name):
    """Get team data with validation."""
    if not team_name:
        return jsonify({'success': False, 'error': 'Team name is required'})
        
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)
    
    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        })
    
    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    """Get player stats with improved error handling."""
    if not player_id:
        return jsonify({'success': False, 'error': 'Player ID is required'})
        
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "tank01-fantasy-stats.p.rapidapi.com"
    }
    
    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo"
    info_querystring = {"playerID": player_id}
    
    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": "2024",
    }
    
    try:
        # Get both responses with timeout
        info_response = requests.get(info_url, headers=headers, params=info_querystring, timeout=10)
        games_response = requests.get(games_url, headers=headers, params=games_querystring, timeout=10)
        
        info_response.raise_for_status()
        games_response.raise_for_status()
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data.get('body', {}).values())
            games.sort(key=lambda x: x.get('gameID', 0), reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data.get('body', {})
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_status = str(player_info['injury'])

            # Combine and return all data
            return jsonify({
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName', 'Unknown'),
                    'position': player_info.get('pos', 'N/A'),
                    'number': player_info.get('jerseyNum', 'N/A'),
                    'height': player_info.get('height', 'N/A'),
                    'weight': player_info.get('weight', 'N/A'),
                    'team': player_info.get('team', 'N/A'),
                    'college': player_info.get('college', 'N/A'),
                    'experience': player_info.get('exp', 'N/A'),
                    'headshot': player_info.get('nbaComHeadshot'),
                    'injury': injury_status
                }
            })
            
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data from API'
        })
        
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'Request timeout'})
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error in player_stats: {str(e)}")
        return jsonify({'success': False, 'error': 'Network error'})
    except Exception as e:
        logger.error(f"Error in player_stats: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

        
team_abbreviations = {
    'Orlando Magic': 'ORL',
    'Minnesota Timberwolves': 'MIN',
    'Miami Heat': 'MIA',
    'Boston Celtics': 'BOS',
    'LA Clippers': 'LAC',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Atlanta Hawks': 'ATL',
    'Cleveland Cavaliers': 'CLE',
    'Toronto Raptors': 'TOR',
    'Washington Wizards': 'WAS',
    'Phoenix Suns': 'PHO',
    'San Antonio Spurs': 'SA',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'Philadelphia 76ers': 'PHI',
    'New Orleans Pelicans': 'NO',
    'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL',
    'Houston Rockets': 'HOU',
    'Brooklyn Nets': 'BKN',
    'New York Knicks': 'NY',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Indiana Pacers': 'IND',
    'Milwaukee Bucks': 'MIL',
    'Golden State Warriors': 'GS',
    'Memphis Grizzlies': 'MEM',
    'Los Angeles Lakers': 'LAL'
}

if __name__ == '__main__':
    app.run(debug=True)