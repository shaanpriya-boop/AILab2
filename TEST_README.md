# Portfolio Advisor Testing Suite

Comprehensive testing framework for the AI-powered Portfolio Advisor system with synthetic data generation, unit tests, integration tests, and end-to-end validation.

## 📁 Test Structure

```
tests/
├── test_synthetic_data.py      # Synthetic data generator
├── test_unit.py                 # Unit tests for components
├── test_integration.py          # Integration workflow tests
├── test_e2e.py                  # End-to-end validation tests
├── run_tests.py                 # Test runner script
├── pytest.ini                   # Pytest configuration
└── requirements-test.txt        # Testing dependencies
```

## 🚀 Quick Start

### 1. Install Testing Dependencies

```bash
pip install -r requirements-test.txt
```

### 2. Generate Synthetic Test Data

```bash
python test_synthetic_data.py
```

This creates `portfolio_test_suite.json` with 20 diverse test cases.

### 3. Run All Tests

```bash
python run_tests.py
```

### 4. Run Specific Test Suites

```bash
# Unit tests only (fast)
python run_tests.py unit

# Integration tests
python run_tests.py integration

# End-to-end tests
python run_tests.py e2e

# Fast tests only (skip slow tests)
python run_tests.py fast

# With coverage report
python run_tests.py coverage
```

## 🧪 Test Categories

### Unit Tests (`test_unit.py`)
Tests individual components in isolation:
- ✅ Prompt generation
- ✅ RAG retrieval quality
- ✅ Agent behavior
- ✅ Portfolio generation logic
- ✅ State management
- ✅ Data validation
- ✅ Error handling

**Run time**: ~5 seconds  
**Command**: `pytest test_unit.py -v`

### Integration Tests (`test_integration.py`)
Tests component interactions:
- ✅ Market analysis workflow
- ✅ LangGraph state propagation
- ✅ API endpoint contracts
- ✅ Observability integration
- ✅ Data persistence
- ✅ Error recovery

**Run time**: ~15 seconds  
**Command**: `pytest test_integration.py -v`

### End-to-End Tests (`test_e2e.py`)
Tests complete user journeys:
- ✅ Complete portfolio generation (low/medium/high risk)
- ✅ Market analysis to portfolio flow
- ✅ Batch processing multiple clients
- ✅ Regression testing
- ✅ Performance benchmarks

**Run time**: ~30 seconds  
**Command**: `pytest test_e2e.py -v -m e2e`

## 📊 Synthetic Data Generator

### Overview
The `SyntheticDataGenerator` creates realistic test data for:
- Client profiles (age, risk appetite, preferences)
- Market conditions (bull/bear/inflation/recovery)
- Boundary rules (risk-based constraints)
- General strategies (Conservative/Balanced/Aggressive)
- Expected outcomes (validation constraints)

### Usage

```python
from test_synthetic_data import SyntheticDataGenerator

# Create generator
generator = SyntheticDataGenerator(seed=42)

# Generate single client
client = generator.generate_client_metadata("CL001")

# Generate market conditions
market = generator.generate_market_conditions("bull_market")

# Generate complete test case
test_case = generator.generate_complete_test_case(
    client_id="TEST001",
    scenario="high_inflation"
)

# Generate test suite
test_suite = generator.generate_test_suite(count=50)
generator.save_test_suite(test_suite, "my_tests.json")
```

### Available Scenarios
- `bull_market` - Strong growth, low inflation
- `bear_market` - Recession, risk-off sentiment
- `high_inflation` - Rising prices, rate hikes
- `recovery` - Economic rebound
- `neutral` - Mixed signals

## ✅ Portfolio Validation

### PortfolioValidator
Validates portfolio outputs against constraints:

```python
from test_e2e import PortfolioValidator

validator = PortfolioValidator()

validation = validator.validate_portfolio(
    portfolio=generated_portfolio,
    constraints=expected_constraints,
    client_metadata=client_data
)

if validation["valid"]:
    print("✓ Portfolio is valid")
else:
    print("✗ Errors:", validation["errors"])
    print("⚠ Warnings:", validation["warnings"])
```

### Validation Checks
1. ✅ Total allocation = 100% (±0.1%)
2. ✅ All 16 asset classes present
3. ✅ Non-negative allocations
4. ✅ Max equity constraint (risk-based)
5. ✅ Min debt constraint (risk-based)
6. ✅ Crypto exclusion (if preferences)
7. ✅ Gold minimum (if preferences)
8. ✅ Excluded assets (zero allocation)
9. ✅ Action validity (BUY/SELL/HOLD logic)

## 📈 Test Markers

Use pytest markers to run specific test types:

```bash
# Run only fast tests
pytest -m fast

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only e2e tests
pytest -m e2e

# Run regression tests
pytest -m regression

# Run performance tests
pytest -m performance
```

## 📊 Coverage Reports

### Generate Coverage Report

```bash
pytest --cov=../ --cov-report=html --cov-report=term-missing
```

### View HTML Report

```bash
open htmlcov/index.html
```

### Coverage Goals
- **Unit Tests**: >80% coverage
- **Critical Paths**: >90% coverage
- **Integration Points**: 100% coverage

## 🔍 Example Test Cases

### Test Low-Risk Portfolio Constraints

```python
def test_low_risk_constraints():
    generator = SyntheticDataGenerator(seed=42)
    test_case = generator.generate_complete_test_case()
    test_case['client_metadata']['risk_appetite'] = 'low'
    
    constraints = generator.generate_expected_portfolio_constraints(
        test_case['client_metadata'],
        test_case['market_conditions']
    )
    
    assert constraints['max_total_equity'] <= 30.0
    assert constraints['min_debt'] >= 40.0
    assert constraints['min_gold'] >= 5.0
```

### Test Market Delta Application

```python
def test_delta_application():
    import pandas as pd
    from marketTrend import apply_instrument_deltas
    
    # Create test strategy
    df = pd.DataFrame([{
        "Client Type": "Test",
        "BankFD": 20.0,
        "Gold": 10.0
    }])
    df.to_csv("test_strategy.csv", index=False)
    
    # Apply deltas
    deltas = {"Gold": 10.0, "BankFD": -5.0}
    result = apply_instrument_deltas(
        "test_strategy.csv",
        "test_output.csv",
        deltas
    )
    
    assert result is not None
```

## 🐛 Debugging Failed Tests

### Verbose Output

```bash
pytest -vv --tb=long test_unit.py::TestClass::test_method
```

### Show Local Variables

```bash
pytest -l test_unit.py
```

### Stop on First Failure

```bash
pytest -x test_unit.py
```

### Run Specific Test

```bash
pytest test_unit.py::TestPromptGeneration::test_market_analysis_questions_generation
```

## 📝 Adding New Tests

### 1. Create Test Class

```python
class TestNewFeature:
    """Test new feature"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_feature_behavior(self, data_generator):
        # Arrange
        test_data = data_generator.generate_client_metadata()
        
        # Act
        result = my_function(test_data)
        
        # Assert
        assert result is not None
```

### 2. Add Test Markers

```python
@pytest.mark.slow
@pytest.mark.integration
def test_expensive_operation():
    pass
```

### 3. Update pytest.ini

Add new markers to `pytest.ini`:

```ini
markers =
    mymarker: description of my marker
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      
      - name: Run unit tests
        run: python run_tests.py unit
      
      - name: Run integration tests
        run: python run_tests.py integration
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 📚 Best Practices

### 1. Test Naming
- Use descriptive names: `test_portfolio_generation_low_risk_client`
- Follow pattern: `test_<what>_<condition>_<expected_result>`

### 2. Test Independence
- Each test should be independent
- Use fixtures for shared setup
- Clean up after tests

### 3. Mock External Dependencies
- Mock LLM calls in unit tests
- Mock API calls
- Mock file I/O when possible

### 4. Use Synthetic Data
- Don't rely on production data
- Generate diverse test cases
- Use reproducible random seeds

### 5. Test Observability
- Verify metrics are collected
- Check guardrail violations
- Validate cost tracking

## 🎯 Test Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| Prompt Generation | 90% | TBD |
| RAG Retrieval | 85% | TBD |
| Agent Logic | 90% | TBD |
| Portfolio Generation | 95% | TBD |
| State Management | 100% | TBD |
| API Endpoints | 90% | TBD |

## 🆘 Troubleshooting

### Tests Failing?

1. **Check dependencies**: `pip install -r requirements-test.txt`
2. **Regenerate test data**: `python test_synthetic_data.py`
3. **Clear cache**: `pytest --cache-clear`
4. **Check Python version**: Requires Python 3.8+

### Slow Tests?

1. **Run fast tests only**: `pytest -m "not slow"`
2. **Use parallel execution**: `pytest -n auto`
3. **Profile tests**: `pytest --durations=10`

### Import Errors?

1. **Add parent to path**: Already handled in test files
2. **Install package in dev mode**: `pip install -e .`

## 📞 Support

For issues or questions:
1. Check test output with `-vv` flag
2. Review test logs in `test_results.xml`
3. Check coverage report in `htmlcov/`

## 🎓 Learning Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Hypothesis for Property Testing](https://hypothesis.readthedocs.io/)

---

**Happy Testing! 🧪✨**
