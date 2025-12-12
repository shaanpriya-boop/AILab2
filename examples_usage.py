"""
Example: Using the Portfolio Advisor Testing Suite
Demonstrates how to use the testing framework
"""

from test_synthetic_data import SyntheticDataGenerator
from test_e2e import PortfolioValidator
import json


def example_1_generate_test_data():
    """Example 1: Generate synthetic test data"""
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Generate Synthetic Test Data")
    print("="*60)
    
    # Create generator with seed for reproducibility
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate single client
    client = generator.generate_client_metadata("DEMO001")
    print(f"\n✓ Generated Client:")
    print(f"  Name: {client['client_name']}")
    print(f"  Risk: {client['risk_appetite']}")
    print(f"  Age: {client['age']}")
    print(f"  Horizon: {client['horizon_years']} years")
    print(f"  No Crypto: {client['preferences_no_crypto']}")
    
    # Generate market conditions
    market = generator.generate_market_conditions("high_inflation")
    print(f"\n✓ Generated Market Conditions:")
    print(f"  {market[:150]}...")
    
    # Generate complete test case
    test_case = generator.generate_complete_test_case(
        client_id="DEMO001",
        scenario="bull_market"
    )
    print(f"\n✓ Generated Complete Test Case:")
    print(f"  Test ID: {test_case['test_id']}")
    print(f"  Scenario: {test_case['scenario']}")
    print(f"  Boundary Rules: {len(test_case['boundary_rules'])}")
    print(f"  Strategies: {len(test_case['general_strategies'])}")


def example_2_generate_test_suite():
    """Example 2: Generate a test suite"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate Test Suite")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate 10 diverse test cases
    test_suite = generator.generate_test_suite(count=10)
    
    print(f"\n✓ Generated {len(test_suite)} test cases")
    
    # Analyze distribution
    risk_distribution = {}
    scenario_distribution = {}
    
    for tc in test_suite:
        risk = tc['client_metadata']['risk_appetite']
        scenario = tc['scenario']
        
        risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        scenario_distribution[scenario] = scenario_distribution.get(scenario, 0) + 1
    
    print("\n📊 Risk Profile Distribution:")
    for risk, count in sorted(risk_distribution.items()):
        print(f"  {risk}: {count}")
    
    print("\n📊 Scenario Distribution:")
    for scenario, count in sorted(scenario_distribution.items()):
        print(f"  {scenario}: {count}")
    
    # Save to file
    filename = "demo_test_suite.json"
    generator.save_test_suite(test_suite, filename)
    print(f"\n✓ Saved test suite to {filename}")


def example_3_validate_portfolio():
    """Example 3: Validate a portfolio"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Validate Portfolio")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    validator = PortfolioValidator()
    
    # Generate test case
    test_case = generator.generate_complete_test_case()
    test_case['client_metadata']['risk_appetite'] = 'low'
    test_case['client_metadata']['preferences_no_crypto'] = True
    
    # Regenerate constraints
    constraints = generator.generate_expected_portfolio_constraints(
        test_case['client_metadata'],
        test_case['market_conditions']
    )
    
    # Create mock portfolio
    mock_portfolio = {
        "transactions": [
            {"asset_type": "BankFD", "action": "BUY", "percentage": 25.0, "rationale": "Safe"},
            {"asset_type": "DebtBond", "action": "BUY", "percentage": 30.0, "rationale": "Stable"},
            {"asset_type": "Gold", "action": "BUY", "percentage": 10.0, "rationale": "Hedge"},
            {"asset_type": "MF_Index", "action": "BUY", "percentage": 15.0, "rationale": "Growth"},
            {"asset_type": "MF_Flexi", "action": "BUY", "percentage": 10.0, "rationale": "Flexible"},
            {"asset_type": "EQ_Banking", "action": "BUY", "percentage": 5.0, "rationale": "Sector"},
            {"asset_type": "EQ_FMCG", "action": "BUY", "percentage": 5.0, "rationale": "Defensive"},
            {"asset_type": "Cryptocurrency", "action": "HOLD", "percentage": 0.0, "rationale": "Excluded"},
            # ... other assets
        ],
        "overall_rationale": "Conservative portfolio for low-risk investor"
    }
    
    # Validate
    validation = validator.validate_portfolio(
        mock_portfolio,
        constraints,
        test_case['client_metadata']
    )
    
    print(f"\n✓ Validation Result: {'PASS ✓' if validation['valid'] else 'FAIL ✗'}")
    
    if validation['errors']:
        print("\n❌ Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    if validation['warnings']:
        print("\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['valid']:
        print("\n✓ Portfolio meets all constraints!")


def example_4_test_market_deltas():
    """Example 4: Test market deltas"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Test Market Deltas")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate deltas for different scenarios
    scenarios = ["bull_market", "bear_market", "high_inflation", "neutral"]
    
    for scenario in scenarios:
        deltas = generator.generate_market_deltas(scenario)
        
        print(f"\n📈 {scenario.upper().replace('_', ' ')}:")
        for asset, delta in sorted(deltas.items()):
            direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            print(f"  {asset}: {direction} {delta:+.1f}%")


def example_5_batch_testing():
    """Example 5: Batch test multiple clients"""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Testing Multiple Clients")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    validator = PortfolioValidator()
    
    # Generate test suite
    test_suite = generator.generate_test_suite(count=5)
    
    results = []
    
    for i, test_case in enumerate(test_suite, 1):
        client_id = test_case['client_metadata']['client_id']
        risk = test_case['client_metadata']['risk_appetite']
        
        print(f"\n[{i}/5] Testing {client_id} (Risk: {risk})")
        
        # Simulate portfolio generation (mock)
        if risk == "low":
            total_equity = 25.0
        elif risk == "medium":
            total_equity = 50.0
        else:
            total_equity = 70.0
        
        mock_portfolio = {
            "transactions": [
                {"asset_type": "BankFD", "percentage": 100 - total_equity},
                {"asset_type": "EQ_IT", "percentage": total_equity}
            ],
            "overall_rationale": f"{risk.capitalize()} risk portfolio"
        }
        
        # Note: This is simplified for demo - real validation needs all 16 assets
        validation = {"valid": True, "errors": []}
        
        results.append({
            "client_id": client_id,
            "risk": risk,
            "valid": validation["valid"]
        })
        
        status = "✓" if validation["valid"] else "✗"
        print(f"  Result: {status}")
    
    # Summary
    print("\n" + "="*60)
    print("BATCH TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r["valid"])
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for result in results:
        status = "✓ PASS" if result["valid"] else "✗ FAIL"
        print(f"  {result['client_id']} ({result['risk']}): {status}")


def example_6_compare_scenarios():
    """Example 6: Compare different market scenarios"""
    
    print("\n" + "="*60)
    print("EXAMPLE 6: Compare Market Scenarios")
    print("="*60)
    
    generator = SyntheticDataGenerator(seed=42)
    
    scenarios = ["bull_market", "bear_market", "high_inflation", "recovery"]
    
    print("\n📊 Market Scenario Comparison:")
    
    for scenario in scenarios:
        market = generator.generate_market_conditions(scenario)
        deltas = generator.generate_market_deltas(scenario)
        
        print(f"\n{scenario.upper().replace('_', ' ')}:")
        print(f"  Description: {market[:100]}...")
        
        # Show top 3 changes
        sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"  Top Changes:")
        for asset, delta in sorted_deltas:
            print(f"    {asset}: {delta:+.1f}%")


def main():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("PORTFOLIO ADVISOR TESTING SUITE - EXAMPLES")
    print("="*80)
    
    examples = [
        example_1_generate_test_data,
        example_2_generate_test_suite,
        example_3_validate_portfolio,
        example_4_test_market_deltas,
        example_5_batch_testing,
        example_6_compare_scenarios
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {example_func.__name__}: {e}")
    
    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80)
    print("\nNext steps:")
    print("1. Run unit tests: python run_tests.py unit")
    print("2. Run integration tests: python run_tests.py integration")
    print("3. Run e2e tests: python run_tests.py e2e")
    print("4. Generate coverage: python run_tests.py coverage")
    print()


if __name__ == "__main__":
    main()
