"""
Data Validation Module for Sports Betting App
Provides comprehensive validation for historical data, API responses, and model inputs.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import requests

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for sports betting application."""
    
    def __init__(self):
        self.required_columns = {
            'games': ['Match Number', 'Date', 'Home Team', 'Away Team'],
            'odds': ['under_over_odds', 'money_line_odds'],
            'team_stats': ['TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L']
        }
        
    def validate_csv_data(self, file_path: str) -> Dict[str, Any]:
        """Validate CSV game data files."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check if file exists and is readable
            df = pd.read_csv(file_path)
            validation_result['stats']['total_rows'] = len(df)
            
            # Check required columns
            missing_columns = [col for col in self.required_columns['games'] if col not in df.columns]
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                validation_result['is_valid'] = False
            
            # Check for empty rows
            empty_rows = df.isnull().all(axis=1).sum()
            if empty_rows > 0:
                validation_result['warnings'].append(f"Found {empty_rows} completely empty rows")
            
            # Validate date format
            if 'Date' in df.columns:
                try:
                    pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='raise')
                except:
                    validation_result['errors'].append("Invalid date format in Date column")
                    validation_result['is_valid'] = False
            
            # Check for duplicate games
            if len(df.columns) >= 4:
                duplicates = df.duplicated(subset=['Date', 'Home Team', 'Away Team']).sum()
                if duplicates > 0:
                    validation_result['warnings'].append(f"Found {duplicates} duplicate games")
            
            # Check for missing results
            if 'Result' in df.columns:
                missing_results = df['Result'].isnull().sum()
                validation_result['stats']['missing_results'] = missing_results
                if missing_results > 0:
                    validation_result['warnings'].append(f"Found {missing_results} games without results")
            
            # Validate team names
            home_teams = df['Home Team'].dropna().unique()
            away_teams = df['Away Team'].dropna().unique()
            all_teams = set(list(home_teams) + list(away_teams))
            validation_result['stats']['unique_teams'] = len(all_teams)
            validation_result['stats']['teams'] = sorted(list(all_teams))
            
        except FileNotFoundError:
            validation_result['errors'].append(f"File not found: {file_path}")
            validation_result['is_valid'] = False
        except pd.errors.EmptyDataError:
            validation_result['errors'].append(f"Empty CSV file: {file_path}")
            validation_result['is_valid'] = False
        except Exception as e:
            validation_result['errors'].append(f"Error reading CSV: {str(e)}")
            validation_result['is_valid'] = False
            
        return validation_result
    
    def validate_sqlite_database(self, db_path: str) -> Dict[str, Any]:
        """Validate SQLite database integrity."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            validation_result['stats']['tables'] = tables
            validation_result['stats']['table_count'] = len(tables)
            
            # Check each table
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    validation_result['stats'][f'{table}_rows'] = row_count
                    
                    # Check for empty tables
                    if row_count == 0:
                        validation_result['warnings'].append(f"Table '{table}' is empty")
                        
                except Exception as e:
                    validation_result['errors'].append(f"Error querying table '{table}': {str(e)}")
            
            conn.close()
            
        except sqlite3.Error as e:
            validation_result['errors'].append(f"SQLite error: {str(e)}")
            validation_result['is_valid'] = False
        except Exception as e:
            validation_result['errors'].append(f"Database validation error: {str(e)}")
            validation_result['is_valid'] = False
            
        return validation_result
    
    def validate_odds_data(self, odds_data: Dict) -> Dict[str, Any]:
        """Validate odds data structure and values."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if not odds_data:
            validation_result['errors'].append("Empty odds data")
            validation_result['is_valid'] = False
            return validation_result
        
        validation_result['stats']['game_count'] = len(odds_data)
        
        for game_key, game_data in odds_data.items():
            try:
                # Validate game key format
                if ':' not in game_key:
                    validation_result['errors'].append(f"Invalid game key format: {game_key}")
                    continue
                
                home_team, away_team = game_key.split(':')
                
                # Check required data structure
                if not isinstance(game_data, dict):
                    validation_result['errors'].append(f"Invalid game data structure for {game_key}")
                    continue
                
                # Validate over/under odds
                if 'under_over_odds' not in game_data:
                    validation_result['warnings'].append(f"Missing over/under odds for {game_key}")
                elif game_data['under_over_odds'] is None:
                    validation_result['warnings'].append(f"Null over/under odds for {game_key}")
                
                # Validate team-specific odds
                for team in [home_team, away_team]:
                    if team not in game_data:
                        validation_result['errors'].append(f"Missing team data for {team} in {game_key}")
                        continue
                    
                    team_data = game_data[team]
                    if not isinstance(team_data, dict):
                        validation_result['errors'].append(f"Invalid team data structure for {team}")
                        continue
                    
                    if 'money_line_odds' not in team_data:
                        validation_result['warnings'].append(f"Missing money line odds for {team}")
                    elif team_data['money_line_odds'] is None:
                        validation_result['warnings'].append(f"Null money line odds for {team}")
                    else:
                        # Validate odds values
                        try:
                            odds_value = int(team_data['money_line_odds'])
                            if abs(odds_value) > 10000:  # Sanity check
                                validation_result['warnings'].append(f"Unusual odds value for {team}: {odds_value}")
                        except (ValueError, TypeError):
                            validation_result['errors'].append(f"Invalid odds format for {team}: {team_data['money_line_odds']}")
                
            except Exception as e:
                validation_result['errors'].append(f"Error validating game {game_key}: {str(e)}")
        
        # Set overall validity
        if validation_result['errors']:
            validation_result['is_valid'] = False
            
        return validation_result
    
    def validate_model_input(self, data: np.ndarray) -> Dict[str, Any]:
        """Validate data before feeding to ML models."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        if data is None:
            validation_result['errors'].append("Input data is None")
            validation_result['is_valid'] = False
            return validation_result
        
        if not isinstance(data, np.ndarray):
            validation_result['errors'].append("Input data is not a numpy array")
            validation_result['is_valid'] = False
            return validation_result
        
        validation_result['stats']['shape'] = data.shape
        validation_result['stats']['dtype'] = str(data.dtype)
        
        # Check for empty data
        if data.size == 0:
            validation_result['errors'].append("Input data is empty")
            validation_result['is_valid'] = False
            return validation_result
        
        # Check for NaN or infinite values
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        
        if nan_count > 0:
            validation_result['errors'].append(f"Found {nan_count} NaN values in input data")
            validation_result['is_valid'] = False
        
        if inf_count > 0:
            validation_result['errors'].append(f"Found {inf_count} infinite values in input data")
            validation_result['is_valid'] = False
        
        # Statistical checks
        validation_result['stats']['mean'] = float(np.mean(data))
        validation_result['stats']['std'] = float(np.std(data))
        validation_result['stats']['min'] = float(np.min(data))
        validation_result['stats']['max'] = float(np.max(data))
        
        # Check for unusual data ranges
        if validation_result['stats']['std'] == 0:
            validation_result['warnings'].append("Zero standard deviation - all values are identical")
        
        if abs(validation_result['stats']['mean']) > 1000:
            validation_result['warnings'].append(f"Unusual mean value: {validation_result['stats']['mean']}")
        
        return validation_result
    
    def validate_historical_data_coverage(self, start_year: int = 2023, end_year: int = 2025) -> Dict[str, Any]:
        """Validate historical data coverage for specified date range."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {},
            'coverage': {}
        }
        
        expected_files = []
        existing_files = []
        
        for year in range(start_year, end_year + 1):
            file_path = f"Data/nba-{year}-UTC.csv"
            expected_files.append(file_path)
            
            try:
                df = pd.read_csv(file_path)
                existing_files.append(file_path)
                
                # Validate year-specific data
                year_validation = self.validate_csv_data(file_path)
                validation_result['coverage'][year] = year_validation
                
                if not year_validation['is_valid']:
                    validation_result['errors'].extend([f"Year {year}: {error}" for error in year_validation['errors']])
                
            except FileNotFoundError:
                validation_result['errors'].append(f"Missing data file for year {year}: {file_path}")
                validation_result['is_valid'] = False
        
        validation_result['stats']['expected_files'] = len(expected_files)
        validation_result['stats']['existing_files'] = len(existing_files)
        validation_result['stats']['missing_files'] = len(expected_files) - len(existing_files)
        
        return validation_result
    
    def cross_validate_with_external_source(self, sample_games: List[Dict]) -> Dict[str, Any]:
        """Cross-validate sample games against external sources (placeholder)."""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # This would implement actual cross-validation with external APIs
        # For now, it's a placeholder that demonstrates the structure
        
        validation_result['warnings'].append("External validation not implemented - manual verification recommended")
        validation_result['stats']['samples_checked'] = len(sample_games)
        
        return validation_result
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report for the entire system."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'validations': {},
            'summary': {
                'total_errors': 0,
                'total_warnings': 0,
                'critical_issues': []
            }
        }
        
        # Validate CSV files
        csv_files = ['Data/nba-2023-UTC.csv', 'Data/nba-2024-UTC.csv']
        for csv_file in csv_files:
            report['validations'][f'csv_{csv_file}'] = self.validate_csv_data(csv_file)
        
        # Validate SQLite databases
        db_files = ['Data/dataset.sqlite', 'Data/TeamData.sqlite', 'Data/OddsData.sqlite']
        for db_file in db_files:
            report['validations'][f'db_{db_file}'] = self.validate_sqlite_database(db_file)
        
        # Validate historical data coverage
        report['validations']['historical_coverage'] = self.validate_historical_data_coverage()
        
        # Calculate summary statistics
        for validation_key, validation_result in report['validations'].items():
            report['summary']['total_errors'] += len(validation_result.get('errors', []))
            report['summary']['total_warnings'] += len(validation_result.get('warnings', []))
            
            if not validation_result.get('is_valid', True):
                report['summary']['critical_issues'].append(validation_key)
        
        # Determine overall status
        if report['summary']['critical_issues']:
            report['overall_status'] = 'critical'
        elif report['summary']['total_errors'] > 0:
            report['overall_status'] = 'errors'
        elif report['summary']['total_warnings'] > 0:
            report['overall_status'] = 'warnings'
        else:
            report['overall_status'] = 'healthy'
        
        return report


def run_full_validation():
    """Run complete data validation and return report."""
    validator = DataValidator()
    report = validator.generate_validation_report()
    
    # Log summary
    logger.info(f"Data validation completed - Status: {report['overall_status']}")
    logger.info(f"Errors: {report['summary']['total_errors']}, Warnings: {report['summary']['total_warnings']}")
    
    if report['summary']['critical_issues']:
        logger.error(f"Critical issues found in: {', '.join(report['summary']['critical_issues'])}")
    
    return report


if __name__ == "__main__":
    # Run validation when script is executed directly
    report = run_full_validation()
    print(f"Validation Status: {report['overall_status']}")
    print(f"Total Errors: {report['summary']['total_errors']}")
    print(f"Total Warnings: {report['summary']['total_warnings']}")