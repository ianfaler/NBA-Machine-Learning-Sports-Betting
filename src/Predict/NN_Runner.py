import copy
import numpy as np
import tensorflow as tf
import logging
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

# Set up logging
logger = logging.getLogger(__name__)

init()

_model = None
_ou_model = None

def _load_models():
    """Load ML models with error handling."""
    global _model, _ou_model
    try:
        if _model is None:
            _model = load_model('Models/NN_Models/Trained-Model-ML-1699315388.285516')
            logger.info("ML model loaded successfully")
        if _ou_model is None:
            _ou_model = load_model("Models/NN_Models/Trained-Model-OU-1699315414.2268295")
            logger.info("O/U model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def nn_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    """Run neural network predictions with improved error handling and validation."""
    if data is None or len(data) == 0:
        logger.error("No data provided for predictions")
        print(Fore.RED + "Error: No valid game data for predictions" + Style.RESET_ALL)
        return
        
    if not games:
        logger.error("No games provided")
        print(Fore.RED + "Error: No games to process" + Style.RESET_ALL)
        return
    
    try:
        _load_models()
        
        # Validate inputs
        if len(games) != len(data):
            logger.warning(f"Mismatch between games ({len(games)}) and data ({len(data)}) lengths")
        
        ml_predictions_array = []

        # Generate ML predictions with error handling
        for i, row in enumerate(data):
            try:
                if row is None or len(row) == 0:
                    logger.warning(f"Invalid data for game {i}")
                    ml_predictions_array.append(np.array([[0.5, 0.5]]))  # Default neutral prediction
                    continue
                    
                prediction = _model.predict(np.array([row]), verbose=0)
                ml_predictions_array.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting ML for game {i}: {e}")
                ml_predictions_array.append(np.array([[0.5, 0.5]]))  # Default neutral prediction

        # Prepare O/U data
        frame_uo = copy.deepcopy(frame_ml)
        
        # Handle missing O/U odds
        processed_ou_odds = []
        for i, ou_odds in enumerate(todays_games_uo):
            if ou_odds is None or ou_odds == '':
                logger.warning(f"Missing O/U odds for game {i}, using default")
                processed_ou_odds.append(220.0)  # Default total
            else:
                try:
                    processed_ou_odds.append(float(ou_odds))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid O/U odds for game {i}: {ou_odds}, using default")
                    processed_ou_odds.append(220.0)
        
        frame_uo['OU'] = np.asarray(processed_ou_odds)
        ou_data = frame_uo.values
        ou_data = ou_data.astype(float)
        ou_data = tf.keras.utils.normalize(ou_data, axis=1)

        ou_predictions_array = []

        # Generate O/U predictions with error handling
        for i, row in enumerate(ou_data):
            try:
                if row is None or len(row) == 0:
                    logger.warning(f"Invalid O/U data for game {i}")
                    ou_predictions_array.append(np.array([[0.5, 0.5]]))  # Default neutral prediction
                    continue
                    
                prediction = _ou_model.predict(np.array([row]), verbose=0)
                ou_predictions_array.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting O/U for game {i}: {e}")
                ou_predictions_array.append(np.array([[0.5, 0.5]]))  # Default neutral prediction

        # Display predictions
        count = 0
        for game in games:
            if count >= len(ml_predictions_array) or count >= len(ou_predictions_array):
                logger.warning(f"Not enough predictions for game {count}")
                break
                
            home_team = game[0]
            away_team = game[1]
            
            try:
                winner = int(np.argmax(ml_predictions_array[count]))
                under_over = int(np.argmax(ou_predictions_array[count]))
                winner_confidence = ml_predictions_array[count]
                un_confidence = ou_predictions_array[count]
                
                # Display predictions with proper formatting
                if winner == 1:
                    winner_confidence_val = round(winner_confidence[0][1] * 100, 1)
                    if under_over == 0:
                        un_confidence_val = round(ou_predictions_array[count][0][0] * 100, 1)
                        print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                              Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(processed_ou_odds[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL)
                    else:
                        un_confidence_val = round(ou_predictions_array[count][0][1] * 100, 1)
                        print(Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                              Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(processed_ou_odds[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL)
                else:
                    winner_confidence_val = round(winner_confidence[0][0] * 100, 1)
                    if under_over == 0:
                        un_confidence_val = round(ou_predictions_array[count][0][0] * 100, 1)
                        print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ': ' +
                              Fore.MAGENTA + 'UNDER ' + Style.RESET_ALL + str(processed_ou_odds[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL)
                    else:
                        un_confidence_val = round(ou_predictions_array[count][0][1] * 100, 1)
                        print(Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence_val}%)" + Style.RESET_ALL + ': ' +
                              Fore.BLUE + 'OVER ' + Style.RESET_ALL + str(processed_ou_odds[count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence_val}%)" + Style.RESET_ALL)
            except Exception as e:
                logger.error(f"Error displaying prediction for game {count}: {e}")
                print(f"Error processing game: {home_team} vs {away_team}")
                
            count += 1

        # Display Expected Values
        if kelly_criterion:
            print("------------Expected Value & Kelly Criterion-----------")
        else:
            print("---------------------Expected Value--------------------")
            
        count = 0
        for game in games:
            if count >= len(ml_predictions_array):
                break
                
            home_team = game[0]
            away_team = game[1]
            ev_home = ev_away = 0
            
            try:
                # Validate odds data
                home_odds = None
                away_odds = None
                
                if count < len(home_team_odds) and home_team_odds[count] is not None:
                    try:
                        home_odds = int(home_team_odds[count])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid home odds for {home_team}: {home_team_odds[count]}")
                        
                if count < len(away_team_odds) and away_team_odds[count] is not None:
                    try:
                        away_odds = int(away_team_odds[count])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid away odds for {away_team}: {away_team_odds[count]}")
                
                # Calculate Expected Values
                if home_odds is not None and away_odds is not None:
                    ev_home = float(Expected_Value.expected_value(ml_predictions_array[count][0][1], home_odds))
                    ev_away = float(Expected_Value.expected_value(ml_predictions_array[count][0][0], away_odds))
                
                expected_value_colors = {
                    'home_color': Fore.GREEN if ev_home > 0 else Fore.RED, 
                    'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
                }
                
                # Kelly Criterion calculations
                bankroll_fraction_home = bankroll_fraction_away = ""
                if kelly_criterion and home_odds is not None and away_odds is not None:
                    try:
                        bankroll_descriptor = ' Fraction of Bankroll: '
                        home_kelly = kc.calculate_kelly_criterion(home_odds, ml_predictions_array[count][0][1])
                        away_kelly = kc.calculate_kelly_criterion(away_odds, ml_predictions_array[count][0][0])
                        bankroll_fraction_home = bankroll_descriptor + str(home_kelly) + '%'
                        bankroll_fraction_away = bankroll_descriptor + str(away_kelly) + '%'
                    except Exception as e:
                        logger.error(f"Error calculating Kelly Criterion: {e}")

                print(f"{home_team} EV: " + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + bankroll_fraction_home)
                print(f"{away_team} EV: " + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + bankroll_fraction_away)
                
            except Exception as e:
                logger.error(f"Error calculating EV for {home_team} vs {away_team}: {e}")
                print(f"Error calculating EV for: {home_team} vs {away_team}")
                
            count += 1

    except Exception as e:
        logger.error(f"Critical error in NN runner: {e}")
        print(Fore.RED + f"Critical error in Neural Network predictions: {e}" + Style.RESET_ALL)
    finally:
        deinit()
