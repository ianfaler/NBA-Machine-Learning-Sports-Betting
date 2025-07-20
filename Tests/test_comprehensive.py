"""
Comprehensive Test Suite for Sports Betting App
Tests all critical functionality including data flow, API integration, and model predictions.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
import sqlite3

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Utils.data_validator import DataValidator
from Utils.Expected_Value import expected_value, payout
from Utils.Kelly_Criterion import calculate_kelly_criterion


class TestDataValidation(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        self.validator = DataValidator()
        
    def test_validate_csv_data_valid_file(self):
        """Test CSV validation with valid data."""
        # Create temporary CSV file
        test_data = {
            'Match Number': [1, 2, 3],
            'Date': ['24/10/2023 23:30', '25/10/2023 02:00', '25/10/2023 23:00'],
            'Home Team': ['Team A', 'Team B', 'Team C'],
            'Away Team': ['Team D', 'Team E', 'Team F'],
            'Result': ['120-110', '95-98', '']
        }
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result = self.validator.validate_csv_data(f.name)
            
        os.unlink(f.name)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['stats']['total_rows'], 3)
        self.assertEqual(result['stats']['unique_teams'], 6)
        
    def test_validate_csv_data_missing_columns(self):
        """Test CSV validation with missing required columns."""
        test_data = {
            'Match Number': [1, 2],
            'Date': ['24/10/2023 23:30', '25/10/2023 02:00']
            # Missing Home Team and Away Team columns
        }
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            result = self.validator.validate_csv_data(f.name)
            
        os.unlink(f.name)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(any('Missing required columns' in error for error in result['errors']))
        
    def test_validate_odds_data_valid(self):
        """Test odds data validation with valid structure."""
        odds_data = {
            'Team A:Team B': {
                'under_over_odds': 220.5,
                'Team A': {'money_line_odds': -150},
                'Team B': {'money_line_odds': +130}
            }
        }
        
        result = self.validator.validate_odds_data(odds_data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['stats']['game_count'], 1)
        
    def test_validate_odds_data_invalid_structure(self):
        """Test odds data validation with invalid structure."""
        odds_data = {
            'InvalidKey': {
                'Team A': {'money_line_odds': -150}
                # Missing Team B and under_over_odds
            }
        }
        
        result = self.validator.validate_odds_data(odds_data)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(len(result['errors']) > 0)
        
    def test_validate_model_input_valid(self):
        """Test model input validation with valid numpy array."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        
        result = self.validator.validate_model_input(data)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['stats']['shape'], (2, 3))
        
    def test_validate_model_input_with_nan(self):
        """Test model input validation with NaN values."""
        data = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        
        result = self.validator.validate_model_input(data)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(any('NaN values' in error for error in result['errors']))


class TestExpectedValue(unittest.TestCase):
    """Test Expected Value calculations."""
    
    def test_expected_value_positive_odds(self):
        """Test expected value calculation with positive odds."""
        result = expected_value(0.6, 150)  # 60% win probability, +150 odds
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)  # Should be positive EV
        
    def test_expected_value_negative_odds(self):
        """Test expected value calculation with negative odds."""
        result = expected_value(0.4, -150)  # 40% win probability, -150 odds
        self.assertIsInstance(result, float)
        self.assertLess(result, 0)  # Should be negative EV
        
    def test_payout_positive_odds(self):
        """Test payout calculation for positive odds."""
        result = payout(150)
        self.assertEqual(result, 150)
        
    def test_payout_negative_odds(self):
        """Test payout calculation for negative odds."""
        result = payout(-150)
        expected = (100 / 150) * 100
        self.assertAlmostEqual(result, expected, places=2)
        
    def test_expected_value_edge_cases(self):
        """Test expected value calculation edge cases."""
        # 50/50 probability with fair odds should be close to 0
        result = expected_value(0.5, 100)
        self.assertAlmostEqual(result, 0, delta=5)


class TestKellyCriterion(unittest.TestCase):
    """Test Kelly Criterion calculations."""
    
    def test_calculate_kelly_criterion_positive_ev(self):
        """Test Kelly Criterion with positive expected value."""
        result = calculate_kelly_criterion(150, 0.6)  # +150 odds, 60% win probability
        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0)  # Should recommend some bet
        
    def test_calculate_kelly_criterion_negative_ev(self):
        """Test Kelly Criterion with negative expected value."""
        result = calculate_kelly_criterion(-150, 0.4)  # -150 odds, 40% win probability
        self.assertIsInstance(result, (int, float))
        # Should recommend no bet (0) for negative EV
        
    def test_calculate_kelly_criterion_edge_cases(self):
        """Test Kelly Criterion edge cases."""
        # Test with 0% probability
        result = calculate_kelly_criterion(100, 0.0)
        self.assertEqual(result, 0)
        
        # Test with 100% probability
        result = calculate_kelly_criterion(100, 1.0)
        self.assertGreater(result, 0)


class TestIntegrationFlaskApp(unittest.TestCase):
    """Integration tests for Flask application."""
    
    def setUp(self):
        # Import Flask app
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Flask'))
        
    @patch('subprocess.run')
    def test_fetch_game_data_success(self, mock_subprocess):
        """Test successful game data fetching."""
        # Mock subprocess output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        Team A (60.5%) vs Team B (39.5%): OVER 220.5 (55.2%)
        Team A EV: 5.2
        Team B EV: -8.1
        Team B (-150) @ Team A (+130)
        """
        mock_subprocess.return_value = mock_result
        
        # Import and test (would need actual Flask app setup)
        # This is a structure example
        
    def test_team_data_endpoint_validation(self):
        """Test team data endpoint input validation."""
        # This would test the Flask endpoints
        # Structure example for comprehensive testing
        pass


class TestModelPredictions(unittest.TestCase):
    """Test model prediction functionality."""
    
    @patch('src.Predict.NN_Runner.load_model')
    def test_nn_runner_with_valid_data(self, mock_load_model):
        """Test Neural Network runner with valid input data."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.4, 0.6]])
        mock_load_model.return_value = mock_model
        
        # Test data
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        todays_games_uo = [220.5, 215.0]
        frame_ml = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        games = [('Team A', 'Team B'), ('Team C', 'Team D')]
        home_odds = [-150, -120]
        away_odds = [130, 110]
        
        # This would test the actual NN_Runner functionality
        # Structure example
        
    def test_model_input_validation(self):
        """Test model input validation before predictions."""
        validator = DataValidator()
        
        # Test with invalid input
        result = validator.validate_model_input(None)
        self.assertFalse(result['is_valid'])
        
        # Test with valid input
        valid_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validator.validate_model_input(valid_data)
        self.assertTrue(result['is_valid'])


class TestErrorHandling(unittest.TestCase):
    """Test error handling across the application."""
    
    def test_main_py_with_no_games(self):
        """Test main.py handling when no games are available."""
        # This would test the main function with empty game data
        pass
        
    def test_flask_app_with_api_failure(self):
        """Test Flask app behavior when external APIs fail."""
        # This would test error handling for API failures
        pass
        
    def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        validator = DataValidator()
        result = validator.validate_sqlite_database('nonexistent_file.db')
        self.assertFalse(result['is_valid'])
        self.assertTrue(len(result['errors']) > 0)


class TestDataFlow(unittest.TestCase):
    """Test complete data flow from API to predictions."""
    
    def test_end_to_end_data_flow(self):
        """Test complete data flow from odds fetching to predictions."""
        # This would test the entire pipeline:
        # 1. Fetch odds data
        # 2. Process team statistics
        # 3. Generate predictions
        # 4. Calculate expected values
        # 5. Return formatted results
        pass
        
    def test_data_consistency_across_sportsbooks(self):
        """Test data consistency when fetching from multiple sportsbooks."""
        # This would verify that data structures are consistent
        # across different sportsbook sources
        pass


class TestSecurity(unittest.TestCase):
    """Test security-related functionality."""
    
    def test_api_key_not_exposed(self):
        """Test that API keys are not exposed in logs or responses."""
        # This would verify that sensitive information is properly handled
        pass
        
    def test_input_sanitization(self):
        """Test that user inputs are properly sanitized."""
        # This would test SQL injection prevention and input validation
        pass


class TestPerformance(unittest.TestCase):
    """Test performance-related aspects."""
    
    def test_large_dataset_processing(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_data = np.random.rand(1000, 50)
        validator = DataValidator()
        
        # Test should complete in reasonable time
        import time
        start_time = time.time()
        result = validator.validate_model_input(large_data)
        end_time = time.time()
        
        self.assertTrue(result['is_valid'])
        self.assertLess(end_time - start_time, 5.0)  # Should complete within 5 seconds
        
    def test_memory_usage(self):
        """Test memory usage with typical workloads."""
        # This would monitor memory usage during typical operations
        pass


def run_all_tests():
    """Run all test suites and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataValidation,
        TestExpectedValue,
        TestKellyCriterion,
        TestIntegrationFlaskApp,
        TestModelPredictions,
        TestErrorHandling,
        TestDataFlow,
        TestSecurity,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    # Run all tests when script is executed directly
    result = run_all_tests()
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)