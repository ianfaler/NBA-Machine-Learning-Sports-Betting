import argparse
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import tensorflow as tf
from colorama import Fore, Style

from src.DataProviders.SbrOddsProvider import SbrOddsProvider
from src.Predict import NN_Runner, XGBoost_Runner
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import create_todays_games_from_odds, get_json_data, to_data_frame, get_todays_games_json, create_todays_games

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

todays_games_url = 'https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/2024/scores/00_todays_scores.json'
data_url = 'https://stats.nba.com/stats/leaguedashteamstats?' \
           'Conference=&DateFrom=&DateTo=&Division=&GameScope=&' \
           'GameSegment=&LastNGames=0&LeagueID=00&Location=&' \
           'MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&' \
           'PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&' \
           'PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&' \
           'Season=2024-25&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&' \
           'StarterBench=&TeamID=0&TwoWay=0&VsConference=&VsDivision='


def createTodaysGames(games, df, odds):
    """Create today's games data with improved error handling and type safety."""
    if not games:
        logger.warning("No games provided")
        return None, [], None, [], []
        
    match_data = []
    todays_games_uo = []
    home_team_odds = []
    away_team_odds = []

    home_team_days_rest = []
    away_team_days_rest = []

    # Use current timestamp for consistent calculations
    current_time = pd.Timestamp.now()

    for game in games:
        home_team = game[0]
        away_team = game[1]
        
        # Validate team names exist in our mapping
        if home_team not in team_index_current or away_team not in team_index_current:
            logger.warning(f"Team not found in index: {home_team} or {away_team}")
            continue
            
        # Get odds data
        if odds is not None:
            game_key = home_team + ':' + away_team
            if game_key in odds:
                game_odds = odds[game_key]
                todays_games_uo.append(game_odds['under_over_odds'])
                home_team_odds.append(game_odds[home_team]['money_line_odds'])
                away_team_odds.append(game_odds[away_team]['money_line_odds'])
            else:
                logger.warning(f"No odds found for game: {game_key}")
                todays_games_uo.append(None)
                home_team_odds.append(None)
                away_team_odds.append(None)
        else:
            todays_games_uo.append(input(home_team + ' vs ' + away_team + ': '))
            home_team_odds.append(input(home_team + ' odds: '))
            away_team_odds.append(input(away_team + ' odds: '))

        # Calculate days rest with proper error handling
        try:
            schedule_df = pd.read_csv('Data/nba-2024-UTC.csv', parse_dates=['Date'], date_format='%d/%m/%Y %H:%M')
        except FileNotFoundError:
            logger.error("NBA schedule file not found")
            home_team_days_rest.append(7)  # Default fallback
            away_team_days_rest.append(7)
            continue
        except Exception as e:
            logger.error(f"Error reading schedule file: {e}")
            home_team_days_rest.append(7)
            away_team_days_rest.append(7)
            continue

        # Get recent games for both teams
        home_games = schedule_df[(schedule_df['Home Team'] == home_team) | (schedule_df['Away Team'] == home_team)]
        away_games = schedule_df[(schedule_df['Home Team'] == away_team) | (schedule_df['Away Team'] == away_team)]
        
        # Find most recent games before today
        previous_home_games = home_games.loc[schedule_df['Date'] <= current_time].sort_values('Date', ascending=False).head(1)
        previous_away_games = away_games.loc[schedule_df['Date'] <= current_time].sort_values('Date', ascending=False).head(1)
        
        # Calculate days off with proper type handling
        if len(previous_home_games) > 0:
            last_home_date = pd.Timestamp(previous_home_games['Date'].iloc[0])
            home_days_off = max(0, (current_time - last_home_date).days + 1)
        else:
            home_days_off = 7  # Default for no recent games
            
        if len(previous_away_games) > 0:
            last_away_date = pd.Timestamp(previous_away_games['Date'].iloc[0])
            away_days_off = max(0, (current_time - last_away_date).days + 1)
        else:
            away_days_off = 7  # Default for no recent games

        home_team_days_rest.append(home_days_off)
        away_team_days_rest.append(away_days_off)
        
        # Get team stats
        try:
            home_team_series = df.iloc[team_index_current.get(home_team)]
            away_team_series = df.iloc[team_index_current.get(away_team)]
            stats = pd.concat([home_team_series, away_team_series])
            stats['Days-Rest-Home'] = home_days_off
            stats['Days-Rest-Away'] = away_days_off
            match_data.append(stats)
        except Exception as e:
            logger.error(f"Error processing team stats for {home_team} vs {away_team}: {e}")
            continue

    if not match_data:
        logger.error("No valid game data processed")
        return None, [], None, [], []

    try:
        games_data_frame = pd.concat(match_data, ignore_index=True, axis=1)
        games_data_frame = games_data_frame.T

        frame_ml = games_data_frame.drop(columns=['TEAM_ID', 'TEAM_NAME'])
        data = frame_ml.values
        data = data.astype(float)

        return data, todays_games_uo, frame_ml, home_team_odds, away_team_odds
    except Exception as e:
        logger.error(f"Error creating final data frame: {e}")
        return None, [], None, [], []


def main():
    """Main function with improved error handling."""
    odds = None
    
    try:
        if args.odds:
            logger.info(f"Fetching odds from {args.odds}")
            odds = SbrOddsProvider(sportsbook=args.odds).get_odds()
            games = create_todays_games_from_odds(odds)
            
            if len(games) == 0:
                logger.warning("No games found from odds provider.")
                return
                
            # Validate odds data alignment
            if games and (games[0][0] + ':' + games[0][1]) not in list(odds.keys()):
                logger.error("Games list not up to date for today's games! Scraping disabled until list is updated.")
                print(Fore.RED + "--------------Games list not up to date for todays games!!! Scraping disabled until list is updated.--------------")
                print(Style.RESET_ALL)
                odds = None
            else:
                print(f"------------------{args.odds} odds data------------------")
                for g in odds.keys():
                    home_team, away_team = g.split(":")
                    print(f"{away_team} ({odds[g][away_team]['money_line_odds']}) @ {home_team} ({odds[g][home_team]['money_line_odds']})")
        else:
            logger.info("Fetching games from NBA API")
            data = get_todays_games_json(todays_games_url)
            games = create_todays_games(data)
            
        # Get team stats data
        logger.info("Fetching team statistics")
        data = get_json_data(data_url)
        df = to_data_frame(data)
        
        # Process games data
        result = createTodaysGames(games, df, odds)
        if result[0] is None:
            logger.error("Failed to process games data")
            return
            
        data, todays_games_uo, frame_ml, home_team_odds, away_team_odds = result
        
        # Run predictions
        if args.nn:
            logger.info("Running Neural Network predictions")
            print("------------Neural Network Model Predictions-----------")
            data_normalized = tf.keras.utils.normalize(data, axis=1)
            NN_Runner.nn_runner(data_normalized, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            
        if args.xgb:
            logger.info("Running XGBoost predictions")
            print("---------------XGBoost Model Predictions---------------")
            XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            
        if args.A:
            logger.info("Running all model predictions")
            print("---------------XGBoost Model Predictions---------------")
            XGBoost_Runner.xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            data_normalized = tf.keras.utils.normalize(data, axis=1)
            print("------------Neural Network Model Predictions-----------")
            NN_Runner.nn_runner(data_normalized, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, args.kc)
            print("-------------------------------------------------------")
            
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model to Run')
    parser.add_argument('-xgb', action='store_true', help='Run with XGBoost Model')
    parser.add_argument('-nn', action='store_true', help='Run with Neural Network Model')
    parser.add_argument('-A', action='store_true', help='Run all Models')
    parser.add_argument('-odds', help='Sportsbook to fetch from. (fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny')
    parser.add_argument('-kc', action='store_true', help='Calculates percentage of bankroll to bet based on model edge')
    args = parser.parse_args()
    main()
