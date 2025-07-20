# 🏀 NBA Sports Betting AI Application

A comprehensive machine learning application for NBA sports betting predictions using Neural Networks and XGBoost models. The application provides real-time odds analysis, expected value calculations, and Kelly Criterion betting recommendations.

## ✨ Features

- **🤖 AI-Powered Predictions**: Neural Network and XGBoost models for game outcome predictions
- **💰 Expected Value Analysis**: Calculate expected value for money line and over/under bets
- **📊 Kelly Criterion**: Optimal bankroll management recommendations
- **🔄 Real-Time Odds**: Integration with multiple sportsbooks (FanDuel, DraftKings, BetMGM)
- **📱 Web Interface**: Responsive Flask web application
- **📈 Historical Data**: Comprehensive NBA data from 2007-2025
- **🛡️ Data Validation**: Robust data integrity checks
- **🧪 Comprehensive Testing**: Full test suite for reliability

## 🚀 Quick Start

### Prerequisites

- Python 3.13+ (recommended) or Python 3.8+
- Git
- Virtual environment support

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd sports-betting-app
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

5. **Run data validation (optional but recommended)**
   ```bash
   python src/Utils/data_validator.py
   ```

6. **Start the Flask web application**
   ```bash
   cd Flask
   python app.py
   ```

7. **Or run command-line predictions**
   ```bash
   # XGBoost predictions with FanDuel odds
   python main.py -xgb -odds=fanduel
   
   # Neural Network predictions
   python main.py -nn -odds=draftkings
   
   # All models with Kelly Criterion
   python main.py -A -odds=betmgm -kc
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# RapidAPI Key for NBA stats
RAPIDAPI_KEY=your_rapidapi_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///Data/dataset.sqlite

# Logging Level
LOG_LEVEL=INFO

# API Rate Limiting (requests per minute)
API_RATE_LIMIT=60

# Cache TTL (Time To Live) in seconds
CACHE_TTL=600
```

### Supported Sportsbooks

- FanDuel (`fanduel`)
- DraftKings (`draftkings`)
- BetMGM (`betmgm`)
- PointsBet (`pointsbet`)
- Caesars (`caesars`)
- Wynn (`wynn`)
- BetRivers NY (`bet_rivers_ny`)

## 📊 Data Sources

### Historical Data
- **NBA Game Data**: 2007-2025 seasons
- **Team Statistics**: Comprehensive team performance metrics
- **Odds Data**: Historical betting odds and outcomes

### Real-Time APIs
- **NBA Stats API**: Live team statistics and game information
- **SBR Scraper**: Real-time sportsbook odds
- **RapidAPI**: Player information and advanced statistics

## 🏗️ Architecture

```
sports-betting-app/
├── main.py                 # Command-line interface
├── Flask/                  # Web application
│   ├── app.py             # Flask server
│   └── templates/         # HTML templates
├── src/                   # Core application
│   ├── DataProviders/     # API integrations
│   ├── Predict/          # ML models
│   └── Utils/            # Utilities and validation
├── Models/               # Trained ML models
│   ├── NN_Models/       # Neural Network models
│   └── XGBoost_Models/  # XGBoost models
├── Data/                # Historical data
├── Tests/               # Test suite
└── requirements.txt     # Dependencies
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python Tests/test_comprehensive.py

# Run specific test categories
python -m pytest Tests/ -v

# Run with coverage report
python -m pytest Tests/ --cov=src --cov-report=html
```

## 🔍 Data Validation

The application includes comprehensive data validation:

```bash
# Run full data validation
python src/Utils/data_validator.py

# Validate specific components
python -c "
from src.Utils.data_validator import DataValidator
validator = DataValidator()
report = validator.generate_validation_report()
print(f'Status: {report[\"overall_status\"]}')
"
```

## 📈 Usage Examples

### Command Line Interface

```bash
# Basic XGBoost prediction
python main.py -xgb

# Neural Network with specific sportsbook
python main.py -nn -odds=fanduel

# All models with Kelly Criterion optimization
python main.py -A -odds=draftkings -kc

# Manual odds input (no API)
python main.py -xgb
```

### Web Interface

1. Start the Flask app: `cd Flask && python app.py`
2. Open browser to `http://localhost:5000`
3. View real-time predictions and odds comparison
4. Click team names for detailed roster information

### API Integration

```python
from src.DataProviders.SbrOddsProvider import SbrOddsProvider

# Fetch odds from FanDuel
provider = SbrOddsProvider(sportsbook="fanduel")
odds = provider.get_odds()

# Process with validation
from src.Utils.data_validator import DataValidator
validator = DataValidator()
result = validator.validate_odds_data(odds)
```

## 🛠️ Recent Fixes & Improvements

### Critical Bug Fixes ✅
- ✅ **Pandas Compatibility**: Updated to pandas>=2.2.0 for Python 3.13 support
- ✅ **Subprocess Issues**: Fixed Python executable detection in Flask app
- ✅ **Regex Patterns**: Corrected confidence parsing regular expressions
- ✅ **Type Safety**: Fixed datetime type mismatches in data processing
- ✅ **HTML Structure**: Resolved nested table row issues in frontend

### Security Enhancements 🔐
- ✅ **API Key Security**: Moved sensitive keys to environment variables
- ✅ **Input Validation**: Added comprehensive input sanitization
- ✅ **Error Handling**: Improved error messages without exposing internals

### Data Flow Improvements 📊
- ✅ **Null Checks**: Added validation for missing data across all components
- ✅ **Data Validation**: Comprehensive validation module for all data sources
- ✅ **Error Recovery**: Graceful handling of API failures and missing data

### User Experience 🎨
- ✅ **Mobile Responsive**: Added mobile-friendly CSS and responsive design
- ✅ **Loading States**: Loading indicators for better user feedback
- ✅ **Error Messages**: User-friendly error displays
- ✅ **Auto-refresh**: Optional page refresh for real-time updates

### Performance & Reliability ⚡
- ✅ **Caching**: Improved caching strategy for API responses
- ✅ **Logging**: Comprehensive logging throughout application
- ✅ **Test Coverage**: Extensive test suite covering all major functionality
- ✅ **Memory Management**: Optimized data processing for large datasets

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Verify all dependencies installed
pip install -r requirements.txt
```

**2. Missing API Keys**
```bash
# Check .env file exists and contains keys
cat .env

# Verify environment variables loaded
python -c "import os; print(os.getenv('RAPIDAPI_KEY'))"
```

**3. Model Loading Errors**
```bash
# Verify model files exist
ls -la Models/NN_Models/
ls -la Models/XGBoost_Models/

# Check file permissions
chmod 644 Models/**/*
```

**4. Database Issues**
```bash
# Validate database integrity
python src/Utils/data_validator.py

# Check file sizes
ls -lh Data/*.sqlite
```

**5. Odds Scraping Failures**
```bash
# Test individual sportsbook
python main.py -xgb -odds=fanduel

# Check network connectivity
ping www.google.com

# Verify SBR scraper version
pip show sbrscrape
```

### Performance Optimization

**Memory Usage**
```bash
# Monitor memory during execution
python -m memory_profiler main.py -xgb

# Optimize pandas usage
export PYTHONMALLOC=malloc
```

**API Rate Limiting**
```python
# Adjust cache TTL in .env
CACHE_TTL=900  # 15 minutes

# Implement backoff strategies
import time
time.sleep(1)  # Between API calls
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python Tests/test_comprehensive.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Sports betting involves risk, and you should never bet more than you can afford to lose. The predictions and recommendations provided by this application are not guaranteed to be accurate. Always gamble responsibly and in accordance with local laws and regulations.

## 🙏 Acknowledgments

- NBA Stats API for comprehensive basketball data
- SBR Scraper for odds data integration
- RapidAPI for player information services
- The open-source machine learning community
- Contributors and testers

## 📞 Support

For support, questions, or feature requests:

1. Check the [troubleshooting section](#-troubleshooting)
2. Run data validation: `python src/Utils/data_validator.py`
3. Check logs in the console output
4. Create an issue with detailed error information

---

**Made with ❤️ for the basketball and machine learning community**
