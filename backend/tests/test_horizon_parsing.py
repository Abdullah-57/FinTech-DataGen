#!/usr/bin/env python3
"""
Unit tests for horizon parsing functionality in FinTech DataGen.

This module tests the _parse_horizon_to_hours function that converts
horizon strings (like '1w', '2m') to hours for forecast calculations.

Author: FinTech DataGen Team
Date: November 2025
"""

import unittest
import sys
import os

# Add parent directory to path to import app functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function from app.py
from app import _parse_horizon_to_hours

class TestHorizonParsing(unittest.TestCase):
    """Test cases for horizon parsing functionality."""
    
    def test_hours_parsing(self):
        """Test parsing of hour-based horizons."""
        test_cases = [
            ('1h', 1),
            ('3h', 3),
            ('24h', 24),
            ('72h', 72),
            ('168h', 168),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected, 
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_days_parsing(self):
        """Test parsing of day-based horizons."""
        test_cases = [
            ('1d', 24),
            ('3d', 72),
            ('7d', 168),
            ('30d', 720),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_weeks_parsing(self):
        """Test parsing of week-based horizons (NEW FEATURE)."""
        test_cases = [
            ('1w', 168),    # 1 week = 7 days = 168 hours
            ('2w', 336),    # 2 weeks = 14 days = 336 hours
            ('4w', 672),    # 4 weeks = 28 days = 672 hours
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_months_parsing(self):
        """Test parsing of month-based horizons (NEW FEATURE)."""
        test_cases = [
            ('1m', 720),    # 1 month = 30 days = 720 hours
            ('2m', 1440),   # 2 months = 60 days = 1440 hours
            ('3m', 2160),   # 3 months = 90 days = 2160 hours
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_case_insensitive(self):
        """Test that parsing is case insensitive."""
        test_cases = [
            ('1H', 1),
            ('1D', 24),
            ('1W', 168),
            ('1M', 720),
            ('2W', 336),
            ('2M', 1440),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        test_cases = [
            (' 1h ', 1),
            (' 2w ', 336),
            (' 1m ', 720),
            ('  3d  ', 72),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        test_cases = [
            ('', 24),           # Empty string -> default
            ('invalid', 24),    # Invalid format -> default
            ('0h', 1),          # Zero -> minimum 1
            ('0w', 1),          # Zero -> minimum 1
            ('0m', 1),          # Zero -> minimum 1
            ('-1h', 1),         # Negative -> minimum 1
            ('abc', 24),        # Non-numeric -> default
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_numeric_only_input(self):
        """Test numeric-only input (assumes hours)."""
        test_cases = [
            ('1', 1),
            ('24', 24),
            ('168', 168),
            ('720', 720),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")
    
    def test_real_world_scenarios(self):
        """Test real-world usage scenarios."""
        # Common forecast horizons used in financial markets
        test_cases = [
            ('1h', 1),      # Intraday trading
            ('4h', 4),      # Short-term analysis
            ('1d', 24),     # Daily forecasts
            ('1w', 168),    # Weekly outlook
            ('2w', 336),    # Bi-weekly analysis
            ('1m', 720),    # Monthly projections
            ('3m', 2160),   # Quarterly forecasts
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = _parse_horizon_to_hours(input_val)
                self.assertEqual(result, expected,
                    f"Expected {expected} hours for '{input_val}', got {result}")

if __name__ == '__main__':
    unittest.main()