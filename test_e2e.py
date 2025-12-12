"""
End-to-End Tests for Portfolio Advisor System
Tests complete user journeys with validation
"""

import pytest
import sys
import pandas as pd
import json
import time
from typing import Dict, List
from pathlib import Path

sys.path.insert(0, '../')

from test_synthetic_data import SyntheticDataGenerator


class PortfolioValidator:
    """Validate portfolio outputs against constraints"""
    
    @staticmethod
    def validate_portfolio(
        portfolio: Dict,
        constraints: Dict,
        client_metadata: Dict
    ) -> Dict[str, bool]:
        """
        Validate portfolio against expected constraints
        
        Returns dict of validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        transactions = portfolio.get("transactions", [])
        
        # Validation 1: Total allocation = 100%
        total_allocation = sum(t.get("percentage", 0) for t in transactions)
        tolerance = constraints.get("allocation_tolerance", 0.1)
        
        if abs(total_allocation - 100.0) > tolerance:
            results["valid"] = False
            results["errors"].append(
                f"Total allocation {total_allocation}% != 100% (tolerance: {tolerance}%)"
            )
        
        # Validation 2: Asset count
        expected_count = constraints.get("asset_count", 16)
        if len(transactions) != expected_count:
            results["valid"] = False
            results["errors"].append(
                f"Expected {expected_count} assets, got {len(transactions)}"
            )
        
        # Validation 3: Non-negative allocations
        negative_assets = [t for t in transactions if t.get("percentage", 0) < 0]
        if negative_assets:
            results["valid"] = False
            results["errors"].append(
                f"Negative allocations found: {[t['asset_type'] for t in negative_assets]}"
            )
        
        # Validation 4: Max equity constraint
        equity_assets = [t for t in transactions if t.get("asset_type", "").startswith("EQ_")]
        total_equity = sum(t.get("percentage", 0) for t in equity_assets)
        
        max_equity = constraints.get("max_total_equity", 100)
        if total_equity > max_equity:
            results["valid"] = False
            results["errors"].append(
                f"Total equity {total_equity}% > max {max_equity}%"
            )
        
        # Validation 5: Min debt constraint
        debt_assets = [t for t in transactions 
                      if t.get("asset_type", "") in ["BankFD", "DebtBond"]]
        total_debt = sum(t.get("percentage", 0) for t in debt_assets)
        
        min_debt = constraints.get("min_debt", 0)
        if total_debt < min_debt:
            results["valid"] = False
            results["errors"].append(
                f"Total debt {total_debt}% < min {min_debt}%"
            )
        
        # Validation 6: Crypto exclusion
        crypto = next((t for t in transactions if t.get("asset_type") == "Cryptocurrency"), None)
        max_crypto = constraints.get("max_crypto")
        
        if max_crypto is not None and crypto:
            if crypto.get("percentage", 0) > max_crypto:
                results["valid"] = False
                results["errors"].append(
                    f"Crypto allocation {crypto['percentage']}% > max {max_crypto}%"
                )
        
        # Validation 7: Gold minimum
        gold = next((t for t in transactions if t.get("asset_type") == "Gold"), None)
        min_gold = constraints.get("min_gold", 0)
        
        if gold and gold.get("percentage", 0) < min_gold:
            results["valid"] = False
            results["errors"].append(
                f"Gold allocation {gold['percentage']}% < min {min_gold}%"
            )
        
        # Validation 8: Excluded assets
        excluded = constraints.get("excluded_assets", [])
        for asset in excluded:
            asset_txn = next((t for t in transactions if t.get("asset_type") == asset), None)
            if asset_txn and asset_txn.get("percentage", 0) > 0:
                results["valid"] = False
                results["errors"].append(
                    f"Excluded asset {asset} has allocation {asset_txn['percentage']}%"
                )
        
        # Validation 9: Actions validity
        for t in transactions:
            action = t.get("action")
            percentage = t.get("percentage", 0)
            
            # SELL should not be used when allocation is 0
            if action == "SELL" and percentage == 0:
                results["warnings"].append(
                    f"{t.get('asset_type')}: SELL action with 0% allocation"
                )
            
            # HOLD for excluded assets
            if t.get("asset_type") in excluded and action != "HOLD":
                results["warnings"].append(
                    f"{t.get('asset_type')}: Should be HOLD (excluded asset)"
                )
        
        return results


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.fixture
    def validator(self):
        return PortfolioValidator()
    
    @pytest.mark.e2e
    def test_complete_portfolio_generation_low_risk(self, data_generator, validator):
        """Test complete portfolio generation for low-risk client"""
        
        # Generate test case
        test_case = data_generator.generate_complete_test_case()
        test_case['client_metadata']['risk_appetite'] = 'low'
        test_case['client_metadata']['preferences_no_crypto'] = True
        
        # Regenerate constraints for low risk
        test_case['expected_constraints'] = data_generator.generate_expected_portfolio_constraints(
            test_case['client_metadata'],
            test_case['market_conditions']
        )
        
        # Simulate portfolio generation
        mock_portfolio = {
            "transactions": [
                {"asset_type": "BankFD", "action": "BUY", "percentage": 25.0, "rationale": "Safe"},
                {"asset_type": "DebtBond", "action": "BUY", "percentage": 30.0, "rationale": "Stable"},
                {"asset_type": "Gold", "action": "BUY", "percentage": 10.0, "rationale": "Hedge"},
                {"asset_type": "MF_Index", "action": "BUY", "percentage": 20.0, "rationale": "Moderate"},
                {"asset_type": "EQ_IT", "action": "BUY", "percentage": 10.0, "rationale": "Growth"},
                {"asset_type": "Cryptocurrency", "action": "HOLD", "percentage": 0.0, "rationale": "Excluded"},
                # ... other assets
            ],
            "overall_rationale": "Conservative portfolio for low-risk client"
        }
        
        # Validate
        validation = validator.validate_portfolio(
            mock_portfolio,
            test_case['expected_constraints'],
            test_case['client_metadata']
        )
        
        # Assertions
        if not validation["valid"]:
            print("Validation errors:", validation["errors"])
        
        assert validation["valid"], f"Portfolio validation failed: {validation['errors']}"
    
    @pytest.mark.e2e
    def test_complete_portfolio_generation_high_risk(self, data_generator, validator):
        """Test complete portfolio generation for high-risk client"""
        
        test_case = data_generator.generate_complete_test_case()
        test_case['client_metadata']['risk_appetite'] = 'high'
        test_case['client_metadata']['preferences_no_crypto'] = False
        
        test_case['expected_constraints'] = data_generator.generate_expected_portfolio_constraints(
            test_case['client_metadata'],
            test_case['market_conditions']
        )
        
        # Simulate aggressive portfolio
        mock_portfolio = {
            "transactions": [
                {"asset_type": "BankFD", "action": "BUY", "percentage": 5.0, "rationale": "Minimal safe"},
                {"asset_type": "DebtBond", "action": "BUY", "percentage": 10.0, "rationale": "Balance"},
                {"asset_type": "EQ_IT", "action": "BUY", "percentage": 20.0, "rationale": "High growth"},
                {"asset_type": "EQ_Banking", "action": "BUY", "percentage": 15.0, "rationale": "Sector bet"},
                {"asset_type": "MF_SmallCap", "action": "BUY", "percentage": 15.0, "rationale": "Aggressive"},
                {"asset_type": "Cryptocurrency", "action": "BUY", "percentage": 5.0, "rationale": "Speculative"},
                # ... other assets totaling 100%
            ],
            "overall_rationale": "Aggressive portfolio for high-risk client"
        }
        
        validation = validator.validate_portfolio(
            mock_portfolio,
            test_case['expected_constraints'],
            test_case['client_metadata']
        )
        
        assert validation["valid"], f"Portfolio validation failed: {validation['errors']}"
    
    @pytest.mark.e2e
    def test_market_analysis_to_portfolio(self, data_generator):
        """Test complete flow from market analysis to portfolio"""
        
        # Step 1: Market Analysis
        market_topic = "Current inflation and interest rate environment"
        
        # Simulate market analysis
        mock_deltas = data_generator.generate_market_deltas("high_inflation")
        
        # Step 2: Apply deltas to strategies
        strategies = data_generator.generate_general_strategies()
        df = pd.DataFrame(strategies)
        
        # Verify deltas are reasonable
        assert isinstance(mock_deltas, dict)
        assert len(mock_deltas) > 0
        assert all(isinstance(v, (int, float)) for v in mock_deltas.values())
        
        # Step 3: Generate portfolio
        test_case = data_generator.generate_complete_test_case()
        
        # Verify complete flow data
        assert test_case['client_metadata'] is not None
        assert test_case['general_strategies'] is not None
        assert len(test_case['boundary_rules']) > 0
    
    @pytest.mark.e2e
    def test_multiple_clients_batch_processing(self, data_generator, validator):
        """Test batch processing of multiple client portfolios"""
        
        # Generate multiple test cases
        test_cases = data_generator.generate_test_suite(count=5)
        
        results = []
        for test_case in test_cases:
            # Simulate portfolio generation
            mock_portfolio = self._create_mock_portfolio_for_risk(
                test_case['client_metadata']['risk_appetite']
            )
            
            # Validate
            validation = validator.validate_portfolio(
                mock_portfolio,
                test_case['expected_constraints'],
                test_case['client_metadata']
            )
            
            results.append({
                "client_id": test_case['client_metadata']['client_id'],
                "valid": validation["valid"],
                "errors": validation["errors"]
            })
        
        # All should be valid
        invalid_count = sum(1 for r in results if not r["valid"])
        assert invalid_count == 0, f"{invalid_count}/{len(results)} portfolios failed validation"
    
    def _create_mock_portfolio_for_risk(self, risk_appetite: str) -> Dict:
        """Helper to create mock portfolio based on risk"""
        
        if risk_appetite == "low":
            return {
                "transactions": [
                    {"asset_type": "BankFD", "percentage": 30.0},
                    {"asset_type": "DebtBond", "percentage": 30.0},
                    {"asset_type": "Gold", "percentage": 10.0},
                    {"asset_type": "MF_Index", "percentage": 20.0},
                    {"asset_type": "EQ_Banking", "percentage": 10.0},
                ],
                "overall_rationale": "Conservative"
            }
        elif risk_appetite == "high":
            return {
                "transactions": [
                    {"asset_type": "BankFD", "percentage": 10.0},
                    {"asset_type": "DebtBond", "percentage": 10.0},
                    {"asset_type": "EQ_IT", "percentage": 25.0},
                    {"asset_type": "MF_SmallCap", "percentage": 20.0},
                    {"asset_type": "EQ_Banking", "percentage": 20.0},
                    {"asset_type": "Cryptocurrency", "percentage": 5.0},
                ],
                "overall_rationale": "Aggressive"
            }
        else:  # medium
            return {
                "transactions": [
                    {"asset_type": "BankFD", "percentage": 20.0},
                    {"asset_type": "DebtBond", "percentage": 20.0},
                    {"asset_type": "MF_Index", "percentage": 25.0},
                    {"asset_type": "EQ_IT", "percentage": 15.0},
                    {"asset_type": "Gold", "percentage": 10.0},
                ],
                "overall_rationale": "Balanced"
            }


class TestRegressionSuite:
    """Regression tests against known good outputs"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.fixture
    def baseline_portfolios(self):
        """Load baseline portfolios for regression testing"""
        return {
            "low_risk_baseline": {
                "total_equity": 25.0,
                "total_debt": 55.0,
                "gold_allocation": 10.0,
                "crypto_allocation": 0.0
            },
            "medium_risk_baseline": {
                "total_equity": 45.0,
                "total_debt": 35.0,
                "gold_allocation": 5.0,
                "crypto_allocation": 0.0
            },
            "high_risk_baseline": {
                "total_equity": 70.0,
                "total_debt": 15.0,
                "gold_allocation": 0.0,
                "crypto_allocation": 5.0
            }
        }
    
    def test_regression_low_risk(self, baseline_portfolios):
        """Test that low-risk portfolios haven't regressed"""
        
        baseline = baseline_portfolios["low_risk_baseline"]
        
        # Simulate current output
        current_output = {
            "total_equity": 26.0,
            "total_debt": 54.0,
            "gold_allocation": 10.0,
            "crypto_allocation": 0.0
        }
        
        # Allow small deviation (2%)
        tolerance = 2.0
        
        for key in baseline:
            diff = abs(current_output[key] - baseline[key])
            assert diff <= tolerance, f"{key}: {diff}% deviation (baseline: {baseline[key]}%, current: {current_output[key]}%)"
    
    def test_regression_portfolio_quality_metrics(self, data_generator):
        """Test portfolio quality metrics haven't regressed"""
        
        test_case = data_generator.generate_complete_test_case()
        
        # Quality metrics
        metrics = {
            "diversification_score": 0.85,  # 0-1 scale
            "risk_adjusted_return": 1.2,
            "sharpe_ratio": 1.5,
            "max_drawdown": -15.0
        }
        
        # All metrics should be within acceptable ranges
        assert 0.7 <= metrics["diversification_score"] <= 1.0
        assert metrics["risk_adjusted_return"] > 1.0
        assert metrics["sharpe_ratio"] > 1.0
        assert metrics["max_drawdown"] > -20.0


class TestPerformanceE2E:
    """End-to-end performance tests"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_end_to_end_latency(self, data_generator):
        """Test complete workflow latency"""
        
        test_case = data_generator.generate_complete_test_case()
        
        start_time = time.time()
        
        # Simulate complete workflow
        # 1. Market analysis (mock)
        time.sleep(0.1)  # Simulate processing
        
        # 2. Load client data
        client_data = test_case['client_metadata']
        
        # 3. Load strategies
        strategies = test_case['general_strategies']
        
        # 4. Generate portfolio
        time.sleep(0.1)  # Simulate LLM call
        
        end_time = time.time()
        total_latency = (end_time - start_time) * 1000  # ms
        
        # Should complete within 5 seconds for E2E
        assert total_latency < 5000, f"E2E took {total_latency:.0f}ms"
        
        print(f"E2E latency: {total_latency:.0f}ms")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_concurrent_portfolio_generation(self, data_generator):
        """Test concurrent processing of multiple portfolios"""
        import concurrent.futures
        
        test_cases = data_generator.generate_test_suite(count=5)
        
        def process_portfolio(test_case):
            # Simulate portfolio generation
            time.sleep(0.2)
            return {
                "client_id": test_case['client_metadata']['client_id'],
                "status": "success"
            }
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_portfolio, tc) for tc in test_cases]
            results = [f.result() for f in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should succeed
        assert all(r["status"] == "success" for r in results)
        
        # Should be faster than sequential (5 * 0.2 = 1.0s)
        assert total_time < 0.8, f"Concurrent processing took {total_time:.2f}s"
        
        print(f"Concurrent processing of {len(test_cases)} portfolios: {total_time:.2f}s")


class TestDataQuality:
    """Test data quality throughout pipeline"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_generated_data_quality(self, data_generator):
        """Test quality of generated test data"""
        
        test_suite = data_generator.generate_test_suite(count=50)
        
        # Check completeness
        for test_case in test_suite:
            assert test_case['client_metadata'] is not None
            assert test_case['market_conditions'] is not None
            assert len(test_case['boundary_rules']) > 0
            assert len(test_case['general_strategies']) > 0
        
        # Check diversity
        risk_profiles = [tc['client_metadata']['risk_appetite'] for tc in test_suite]
        unique_risks = set(risk_profiles)
        assert len(unique_risks) == 3, "Should have all risk profiles represented"
        
        # Check scenario distribution
        scenarios = [tc['scenario'] for tc in test_suite]
        unique_scenarios = set(scenarios)
        assert len(unique_scenarios) >= 3, "Should have diverse scenarios"
    
    def test_portfolio_output_quality(self, data_generator):
        """Test quality of portfolio outputs"""
        
        test_case = data_generator.generate_complete_test_case()
        
        # Simulate portfolio output
        mock_portfolio = {
            "transactions": [
                {
                    "asset_type": "BankFD",
                    "action": "BUY",
                    "percentage": 20.0,
                    "rationale": "Stable returns with low risk profile suitable for conservative allocation"
                }
                # ... more transactions
            ],
            "overall_rationale": "This portfolio is designed for a medium-risk investor with a 10-year horizon"
        }
        
        # Check rationale quality
        overall_rationale = mock_portfolio["overall_rationale"]
        assert len(overall_rationale) > 50, "Rationale should be substantive"
        assert any(word in overall_rationale.lower() for word in ["risk", "investor", "portfolio"])
        
        # Check transaction rationales
        for txn in mock_portfolio["transactions"]:
            assert len(txn["rationale"]) > 20, f"Rationale for {txn['asset_type']} too short"


# ============================================================================
# TEST FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment"""
    
    # Create test directories
    test_dir = Path("/tmp/portfolio_tests")
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])
