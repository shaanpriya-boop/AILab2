"""
Synthetic Data Generator for Portfolio Advisor Testing
Generates realistic test data for clients, market conditions, and expected outcomes
"""

import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json


class SyntheticDataGenerator:
    """Generate synthetic data for testing portfolio advisor system"""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        random.seed(seed)
        self.seed = seed
    
    # ========================================================================
    # CLIENT DATA GENERATION
    # ========================================================================
    
    def generate_client_metadata(self, client_id: str = None) -> Dict:
        """Generate synthetic client metadata"""
        if client_id is None:
            client_id = f"CL{random.randint(1, 999):03d}"
        
        risk_appetites = ["low", "medium", "high"]
        
        return {
            "client_id": client_id,
            "client_name": self._generate_name(),
            "age": random.randint(25, 70),
            "risk_appetite": random.choice(risk_appetites),
            "horizon_years": random.randint(1, 30),
            "liquidity_need": random.choice(["low", "medium", "high"]),
            "preferences_no_crypto": random.choice([True, False]),
            "preferences_min_gold_percent": random.choice([0, 5, 10]),
            "preferences_asset_exclusions": random.sample(
                ["Cryptocurrency", "RealEstate"], 
                k=random.randint(0, 2)
            ) if random.random() > 0.7 else [],
            "annual_income": random.randint(500000, 10000000),
            "investment_amount": random.randint(100000, 50000000)
        }
    
    def generate_multiple_clients(self, count: int = 10) -> List[Dict]:
        """Generate multiple client profiles"""
        return [self.generate_client_metadata(f"TEST{i:03d}") for i in range(count)]
    
    # ========================================================================
    # MARKET CONDITIONS GENERATION
    # ========================================================================
    
    def generate_market_conditions(self, scenario: str = None) -> str:
        """Generate market condition descriptions"""
        
        scenarios = {
            "bull_market": [
                "Strong economic growth with low inflation. Equity markets showing sustained upward momentum. "
                "Corporate earnings beating expectations. Technology and financial sectors leading gains. "
                "Interest rates stable with accommodative monetary policy.",
                
                "Bull market conditions prevail with S&P 500 at all-time highs. GDP growth exceeding 3%. "
                "Unemployment at historic lows. Consumer confidence strong. Low volatility environment. "
                "Central banks maintaining supportive stance."
            ],
            
            "bear_market": [
                "Economic recession concerns mounting. Equity markets in correction territory with 20%+ decline. "
                "Rising unemployment and weakening consumer spending. Corporate earnings under pressure. "
                "Flight to safety driving bond and gold prices higher.",
                
                "Bear market conditions with high volatility. Risk-off sentiment dominating. Credit spreads widening. "
                "Defensive sectors outperforming. Central banks raising rates aggressively to combat inflation. "
                "Geopolitical tensions adding to market stress."
            ],
            
            "high_inflation": [
                "Inflation running at 7%+ levels. Central banks hiking interest rates aggressively. "
                "Commodity prices elevated. Real estate showing weakness. Gold and inflation-protected securities in demand. "
                "Equity valuations compressing under rate pressure.",
                
                "Persistent inflation concerns. Energy and food prices surging. Wage inflation accelerating. "
                "Central banks committed to restrictive policy. Bond yields rising sharply. "
                "Commodities and real assets preferred as inflation hedges."
            ],
            
            "recovery": [
                "Economic recovery gaining traction after slowdown. Equity markets rebounding from lows. "
                "Central banks pausing rate hikes. Corporate earnings stabilizing. "
                "Cyclical sectors beginning to outperform defensives.",
                
                "Early recovery phase with improving economic indicators. Manufacturing activity expanding. "
                "Services sector showing strength. Equity valuations attractive after correction. "
                "Credit conditions easing. Risk appetite gradually returning."
            ],
            
            "neutral": [
                "Mixed economic signals. Markets trading in range-bound pattern. "
                "Moderate inflation with stable interest rates. Balanced sector performance. "
                "Investors adopting wait-and-see approach amid uncertainty.",
                
                "Neutral market environment with low volatility. Economic data showing neither strength nor weakness. "
                "Central banks maintaining steady policy stance. Diversified portfolios performing in-line. "
                "No clear directional trend in major asset classes."
            ]
        }
        
        if scenario and scenario in scenarios:
            return random.choice(scenarios[scenario])
        else:
            # Random scenario
            all_scenarios = [desc for scenario_list in scenarios.values() for desc in scenario_list]
            return random.choice(all_scenarios)
    
    # ========================================================================
    # BOUNDARY RULES GENERATION
    # ========================================================================
    
    def generate_boundary_rules(self, risk_appetite: str) -> List[str]:
        """Generate boundary rules based on risk profile"""
        
        rules_by_risk = {
            "low": [
                "max_equity_for_low_risk: Maximum 30% allocation to equities",
                "crypto_cap: Cryptocurrency allocation limited to 0% for low risk",
                "gold_corridor: Minimum 10% allocation to gold for safety",
                "debt_minimum: Minimum 50% allocation to debt instruments",
                "liquidity_requirement: Maintain minimum 15% in highly liquid assets"
            ],
            "medium": [
                "max_equity_for_medium_risk: Maximum 60% allocation to equities",
                "crypto_cap: Cryptocurrency allocation limited to 5%",
                "gold_corridor: Maintain 5-15% allocation to gold",
                "balanced_portfolio: Maintain balance between growth and safety",
                "sector_diversification: No single equity sector above 15%"
            ],
            "high": [
                "max_equity_for_high_risk: Maximum 80% allocation to equities",
                "crypto_allowance: Cryptocurrency allocation up to 10%",
                "growth_focus: Prioritize high-growth assets",
                "aggressive_allocation: Can concentrate in high-conviction sectors",
                "gold_optional: Gold allocation optional, minimum 0%"
            ]
        }
        
        return rules_by_risk.get(risk_appetite, rules_by_risk["medium"])
    
    # ========================================================================
    # STRATEGY DATA GENERATION
    # ========================================================================
    
    def generate_general_strategies(self) -> List[Dict]:
        """Generate general strategy templates"""
        
        strategies = [
            {
                "Client Type": "Conservative",
                "BankFD": 25.0,
                "DebtBond": 30.0,
                "MF_Index": 15.0,
                "MF_Flexi": 10.0,
                "MF_SmallCap": 0.0,
                "EQ_Banking": 5.0,
                "EQ_Automobile": 0.0,
                "EQ_IT": 0.0,
                "EQ_FMCG": 5.0,
                "EQ_MetalsMining": 0.0,
                "EQ_OilGas": 0.0,
                "EQ_Pharma": 5.0,
                "EQ_Defense": 0.0,
                "Gold": 5.0,
                "Silver": 0.0,
                "RealEstate": 0.0,
                "Cryptocurrency": 0.0
            },
            {
                "Client Type": "Balanced",
                "BankFD": 15.0,
                "DebtBond": 20.0,
                "MF_Index": 20.0,
                "MF_Flexi": 15.0,
                "MF_SmallCap": 5.0,
                "EQ_Banking": 5.0,
                "EQ_Automobile": 3.0,
                "EQ_IT": 5.0,
                "EQ_FMCG": 3.0,
                "EQ_MetalsMining": 2.0,
                "EQ_OilGas": 2.0,
                "EQ_Pharma": 3.0,
                "EQ_Defense": 0.0,
                "Gold": 2.0,
                "Silver": 0.0,
                "RealEstate": 0.0,
                "Cryptocurrency": 0.0
            },
            {
                "Client Type": "Aggressive",
                "BankFD": 5.0,
                "DebtBond": 10.0,
                "MF_Index": 15.0,
                "MF_Flexi": 15.0,
                "MF_SmallCap": 10.0,
                "EQ_Banking": 8.0,
                "EQ_Automobile": 5.0,
                "EQ_IT": 10.0,
                "EQ_FMCG": 5.0,
                "EQ_MetalsMining": 5.0,
                "EQ_OilGas": 5.0,
                "EQ_Pharma": 5.0,
                "EQ_Defense": 2.0,
                "Gold": 0.0,
                "Silver": 0.0,
                "RealEstate": 0.0,
                "Cryptocurrency": 0.0
            },
            {
                "Client Type": "Growth",
                "BankFD": 10.0,
                "DebtBond": 15.0,
                "MF_Index": 25.0,
                "MF_Flexi": 20.0,
                "MF_SmallCap": 10.0,
                "EQ_Banking": 5.0,
                "EQ_Automobile": 3.0,
                "EQ_IT": 7.0,
                "EQ_FMCG": 3.0,
                "EQ_MetalsMining": 0.0,
                "EQ_OilGas": 0.0,
                "EQ_Pharma": 2.0,
                "EQ_Defense": 0.0,
                "Gold": 0.0,
                "Silver": 0.0,
                "RealEstate": 0.0,
                "Cryptocurrency": 0.0
            }
        ]
        
        return strategies
    
    # ========================================================================
    # EXPECTED OUTCOMES GENERATION
    # ========================================================================
    
    def generate_expected_portfolio_constraints(
        self, 
        client_metadata: Dict, 
        market_conditions: str
    ) -> Dict:
        """Generate expected constraints for portfolio validation"""
        
        constraints = {
            "total_allocation": 100.0,
            "allocation_tolerance": 0.1,  # Allow 0.1% deviation
            "asset_count": 16,  # All asset classes must be present
        }
        
        # Risk-based constraints
        if client_metadata["risk_appetite"] == "low":
            constraints["max_total_equity"] = 30.0
            constraints["min_debt"] = 40.0
            constraints["min_gold"] = 5.0
        elif client_metadata["risk_appetite"] == "medium":
            constraints["max_total_equity"] = 60.0
            constraints["min_debt"] = 20.0
        else:  # high risk
            constraints["max_total_equity"] = 85.0
        
        # Crypto constraints
        if client_metadata.get("preferences_no_crypto"):
            constraints["max_crypto"] = 0.0
        
        # Gold constraints
        if client_metadata.get("preferences_min_gold_percent", 0) > 0:
            constraints["min_gold"] = client_metadata["preferences_min_gold_percent"]
        
        # Excluded assets
        if client_metadata.get("preferences_asset_exclusions"):
            constraints["excluded_assets"] = client_metadata["preferences_asset_exclusions"]
        
        return constraints
    
    # ========================================================================
    # NEWS ARTICLES GENERATION (for market analysis)
    # ========================================================================
    
    def generate_news_articles(self, topic: str, count: int = 5) -> List[str]:
        """Generate synthetic news articles for market analysis testing"""
        
        templates = [
            "Central banks signal {action} in response to {indicator}. Market analysts predict {outcome}.",
            "Corporate earnings {performance} expectations with {sector} leading the way. Investors {reaction}.",
            "Inflation data shows {trend} as consumer prices {movement}. Fed officials {statement}.",
            "Equity markets {direction} amid {event}. Volatility index {volatility_change}.",
            "Commodity prices {commodity_trend} with {commodity} seeing significant {price_change}."
        ]
        
        variables = {
            "action": ["rate hikes", "rate cuts", "policy tightening", "accommodative stance"],
            "indicator": ["inflation concerns", "economic growth", "employment data", "GDP figures"],
            "outcome": ["continued volatility", "market stabilization", "sector rotation", "risk-off sentiment"],
            "performance": ["beat", "miss", "meet", "exceed"],
            "sector": ["technology", "financials", "healthcare", "energy"],
            "reaction": ["remain cautious", "increase allocations", "rotate to defensives", "seek opportunities"],
            "trend": ["acceleration", "moderation", "persistent elevation", "decline"],
            "movement": ["rise sharply", "ease", "remain elevated", "stabilize"],
            "statement": ["remain data-dependent", "signal further action", "maintain guidance", "revise outlook"],
            "direction": ["rally", "decline", "consolidate", "reach new highs"],
            "event": ["geopolitical tensions", "economic data releases", "corporate guidance", "policy changes"],
            "volatility_change": ["spikes", "subsides", "remains elevated", "normalizes"],
            "commodity_trend": ["surge", "decline", "stabilize", "show mixed signals"],
            "commodity": ["oil", "gold", "copper", "agricultural products"],
            "price_change": ["gains", "losses", "volatility", "momentum"]
        }
        
        articles = []
        for _ in range(count):
            template = random.choice(templates)
            article = template.format(
                **{key: random.choice(values) for key, values in variables.items()}
            )
            articles.append(article)
        
        return articles
    
    # ========================================================================
    # MARKET DELTAS GENERATION
    # ========================================================================
    
    def generate_market_deltas(self, scenario: str = "neutral") -> Dict[str, float]:
        """Generate market condition deltas for strategy adjustment"""
        
        delta_scenarios = {
            "bull_market": {
                "BankFD": -10.0,
                "DebtBond": -5.0,
                "MF_Index": 5.0,
                "MF_Flexi": 5.0,
                "MF_SmallCap": 3.0,
                "EQ_Banking": 2.0,
                "EQ_IT": 5.0,
                "Gold": -5.0
            },
            "bear_market": {
                "BankFD": 10.0,
                "DebtBond": 10.0,
                "MF_Index": -5.0,
                "EQ_Banking": -5.0,
                "EQ_IT": -8.0,
                "Gold": 10.0
            },
            "high_inflation": {
                "BankFD": -5.0,
                "DebtBond": -10.0,
                "Gold": 15.0,
                "RealEstate": 5.0,
                "EQ_MetalsMining": 5.0
            },
            "neutral": {
                # Small random adjustments
                asset: random.uniform(-2.0, 2.0) 
                for asset in ["BankFD", "DebtBond", "MF_Index", "Gold"]
            }
        }
        
        return delta_scenarios.get(scenario, delta_scenarios["neutral"])
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_name(self) -> str:
        """Generate random client name"""
        first_names = ["Raj", "Priya", "Amit", "Anjali", "Vikram", "Sneha", 
                      "Rahul", "Pooja", "Arjun", "Neha", "Karthik", "Divya"]
        last_names = ["Sharma", "Patel", "Kumar", "Singh", "Reddy", "Iyer",
                     "Gupta", "Rao", "Nair", "Mehta", "Desai", "Pillai"]
        
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    # ========================================================================
    # TEST CASE GENERATION
    # ========================================================================
    
    def generate_complete_test_case(
        self, 
        client_id: str = None,
        scenario: str = None
    ) -> Dict:
        """Generate a complete test case with all required data"""
        
        client_metadata = self.generate_client_metadata(client_id)
        market_conditions = self.generate_market_conditions(scenario)
        boundary_rules = self.generate_boundary_rules(client_metadata["risk_appetite"])
        general_strategies = self.generate_general_strategies()
        expected_constraints = self.generate_expected_portfolio_constraints(
            client_metadata, 
            market_conditions
        )
        
        return {
            "test_id": f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            "client_metadata": client_metadata,
            "market_conditions": market_conditions,
            "boundary_rules": boundary_rules,
            "general_strategies": general_strategies,
            "expected_constraints": expected_constraints,
            "scenario": scenario or "neutral"
        }
    
    def generate_test_suite(self, count: int = 20) -> List[Dict]:
        """Generate a complete test suite with diverse scenarios"""
        
        scenarios = ["bull_market", "bear_market", "high_inflation", "recovery", "neutral"]
        test_suite = []
        
        for i in range(count):
            scenario = scenarios[i % len(scenarios)]
            test_case = self.generate_complete_test_case(
                client_id=f"TEST{i:03d}",
                scenario=scenario
            )
            test_suite.append(test_case)
        
        return test_suite
    
    def save_test_suite(self, test_suite: List[Dict], filename: str = "test_suite.json"):
        """Save test suite to JSON file"""
        with open(filename, 'w') as f:
            json.dump(test_suite, f, indent=2)
        print(f"Test suite saved to {filename}")
    
    def load_test_suite(self, filename: str = "test_suite.json") -> List[Dict]:
        """Load test suite from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create generator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate single test case
    print("="*60)
    print("SINGLE TEST CASE")
    print("="*60)
    test_case = generator.generate_complete_test_case(scenario="bull_market")
    print(f"Test ID: {test_case['test_id']}")
    print(f"Client: {test_case['client_metadata']['client_name']}")
    print(f"Risk: {test_case['client_metadata']['risk_appetite']}")
    print(f"Scenario: {test_case['scenario']}")
    print(f"Market: {test_case['market_conditions'][:100]}...")
    print()
    
    # Generate test suite
    print("="*60)
    print("GENERATING TEST SUITE")
    print("="*60)
    test_suite = generator.generate_test_suite(count=20)
    generator.save_test_suite(test_suite, "portfolio_test_suite.json")
    print(f"Generated {len(test_suite)} test cases")
    print()
    
    # Show distribution
    scenarios = {}
    for tc in test_suite:
        scenario = tc['scenario']
        scenarios[scenario] = scenarios.get(scenario, 0) + 1
    
    print("Scenario Distribution:")
    for scenario, count in scenarios.items():
        print(f"  {scenario}: {count}")
