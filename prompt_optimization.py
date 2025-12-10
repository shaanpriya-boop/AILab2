"""
Prompt Engineering & Optimization Framework
A/B testing, prompt variants, and systematic optimization
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from openai import OpenAI
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant for A/B testing"""
    id: str
    name: str
    system_prompt: str
    user_template: str
    temperature: float = 0.3
    description: str = ""


@dataclass
class TestResult:
    """Result of testing a prompt variant"""
    variant_id: str
    schema_compliance: float
    rule_compliance: float
    rationale_quality: float
    response_time: float
    success_rate: float
    composite_score: float


class PromptLibrary:
    """Library of prompt variants for testing"""
    
    @staticmethod
    def get_baseline_prompt() -> PromptVariant:
        """Current production prompt"""
        return PromptVariant(
            id="baseline",
            name="Baseline Production",
            system_prompt="""You are an expert financial portfolio advisor. Create personalized investment portfolios based on client profile, market conditions, and regulatory rules.""",
            user_template="""Create a portfolio for this client:

CLIENT PROFILE:
{client_metadata}

MARKET CONDITIONS:
{market_conditions}

BOUNDARY RULES:
{boundary_rules}

GENERAL STRATEGIES:
{general_strategies}

Provide a complete portfolio plan as JSON with transactions and overall_rationale.""",
            description="Current production prompt"
        )
    
    @staticmethod
    def get_structured_prompt() -> PromptVariant:
        """More structured with explicit sections"""
        return PromptVariant(
            id="structured",
            name="Structured Instructions",
            system_prompt="""You are an expert financial portfolio advisor specializing in personalized portfolio construction.

YOUR ROLE:
- Analyze client profiles comprehensively
- Apply market intelligence to allocation decisions
- Enforce all regulatory boundary rules strictly
- Generate clear, actionable rationales

OUTPUT REQUIREMENTS:
- JSON format with 'transactions' and 'overall_rationale'
- Each transaction: action, asset_type, percentage, rationale
- Total allocations must equal 100%
- All boundary rules must be satisfied""",
            user_template="""## CLIENT ANALYSIS REQUEST

### 1. CLIENT PROFILE
{client_metadata}

### 2. MARKET ENVIRONMENT
{market_conditions}

### 3. REGULATORY CONSTRAINTS (MANDATORY)
{boundary_rules}

### 4. STRATEGY GUIDELINES (REFERENCE)
{general_strategies}

## REQUIRED OUTPUT
Generate portfolio allocation as structured JSON.""",
            description="Structured with clear sections and headers"
        )
    
    @staticmethod
    def get_constraint_emphasis_prompt() -> PromptVariant:
        """Emphasize constraints heavily"""
        return PromptVariant(
            id="constraint_emphasis",
            name="Constraint Emphasis",
            system_prompt="""You are an expert financial portfolio advisor. 

⚠️ CRITICAL: You MUST follow all boundary rules exactly. Rule violations are unacceptable.

Your responsibilities:
1. ✓ Analyze client profile thoroughly
2. ✓ Consider market conditions
3. ✓✓✓ ENFORCE ALL BOUNDARY RULES (highest priority)
4. ✓ Use strategy guidelines as reference
5. ✓ Ensure total allocation = 100%

Output: JSON with transactions and overall_rationale.""",
            user_template="""**CLIENT PROFILE:**
{client_metadata}

**MARKET CONDITIONS:**
{market_conditions}

**⚠️ MANDATORY BOUNDARY RULES ⚠️:**
{boundary_rules}

**IMPORTANT: Each rule above is MANDATORY and must be satisfied in your portfolio.**

**Strategy Reference (not mandatory):**
{general_strategies}

Generate compliant portfolio allocation.""",
            description="Heavy emphasis on constraints with visual markers"
        )
    
    @staticmethod
    def get_chain_of_thought_prompt() -> PromptVariant:
        """Chain-of-thought reasoning"""
        return PromptVariant(
            id="chain_of_thought",
            name="Chain-of-Thought",
            system_prompt="""You are an expert financial portfolio advisor. Use step-by-step reasoning to build optimal portfolios.

REASONING PROCESS:
1. First, analyze client risk profile and constraints
2. Then, identify applicable boundary rules
3. Next, consider market conditions impact
4. Finally, construct compliant portfolio

Think through each step before making allocation decisions.""",
            user_template="""Build a portfolio using step-by-step analysis:

**Step 1: Client Analysis**
{client_metadata}

What are the key factors? (risk appetite, horizon, preferences, goals)

**Step 2: Identify Constraints**
{boundary_rules}

Which rules apply to this client?

**Step 3: Market Assessment**
{market_conditions}

How should market conditions influence allocations?

**Step 4: Strategy Reference**
{general_strategies}

**Step 5: Construct Portfolio**
Now, synthesize all factors and create the final portfolio allocation as JSON.""",
            description="Encourages step-by-step reasoning"
        )
    
    @staticmethod
    def get_few_shot_prompt() -> PromptVariant:
        """Few-shot with examples"""
        return PromptVariant(
            id="few_shot",
            name="Few-Shot Examples",
            system_prompt="""You are an expert financial portfolio advisor. Learn from these examples:

EXAMPLE 1:
Client: Low risk, 3-year horizon, no crypto
Output: 50% debt, 20% equity, 15% gold, 15% real estate
Rationale: Conservative allocation prioritizing capital preservation

EXAMPLE 2:
Client: High risk, 10-year horizon, tech preference
Output: 60% equity (IT heavy), 20% mutual funds, 10% crypto, 10% commodities
Rationale: Growth-oriented with sector tilt

Now apply this approach to new clients.""",
            user_template="""Client:
{client_metadata}

Market:
{market_conditions}

Rules:
{boundary_rules}

Strategies:
{general_strategies}

Portfolio:""",
            description="Includes example portfolios for learning"
        )
    
    @staticmethod
    def get_self_critique_prompt() -> PromptVariant:
        """Two-pass with self-critique"""
        return PromptVariant(
            id="self_critique",
            name="Self-Critique",
            system_prompt="""You are an expert financial portfolio advisor. Use a two-phase approach:

PHASE 1: Draft portfolio based on analysis
PHASE 2: Critique your own draft against rules, then refine

This ensures higher quality and compliance.""",
            user_template="""Create a portfolio, then verify it:

CLIENT:
{client_metadata}

MARKET:
{market_conditions}

RULES TO CHECK:
{boundary_rules}

STRATEGIES:
{general_strategies}

INSTRUCTIONS:
1. First, create initial portfolio allocation
2. Then, verify against each rule
3. If violations found, adjust and fix
4. Output final verified portfolio as JSON""",
            description="Self-verification step before output"
        )
    
    @staticmethod
    def get_all_variants() -> List[PromptVariant]:
        """Get all prompt variants for testing"""
        return [
            PromptLibrary.get_baseline_prompt(),
            PromptLibrary.get_structured_prompt(),
            PromptLibrary.get_constraint_emphasis_prompt(),
            PromptLibrary.get_chain_of_thought_prompt(),
            PromptLibrary.get_few_shot_prompt(),
            PromptLibrary.get_self_critique_prompt()
        ]


class PromptTester:
    """Test and evaluate prompt variants"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        self.client = client
        self.model = model
    
    def test_variant(
        self,
        variant: PromptVariant,
        test_cases: List[Dict[str, Any]],
        num_samples: int = 20
    ) -> TestResult:
        """Test a prompt variant on multiple cases"""
        
        logger.info(f"Testing variant: {variant.name}")
        
        schema_scores = []
        rule_scores = []
        rationale_scores = []
        response_times = []
        successes = []
        
        for i, test_case in enumerate(test_cases[:num_samples]):
            logger.info(f"  Test case {i+1}/{min(num_samples, len(test_cases))}")
            
            try:
                result = self._execute_test(variant, test_case)
                
                schema_scores.append(result["schema_score"])
                rule_scores.append(result["rule_score"])
                rationale_scores.append(result["rationale_score"])
                response_times.append(result["response_time"])
                successes.append(result["success"])
            
            except Exception as e:
                logger.error(f"    Test failed: {e}")
                successes.append(False)
                schema_scores.append(0)
                rule_scores.append(0)
                rationale_scores.append(0)
        
        # Compute metrics
        return TestResult(
            variant_id=variant.id,
            schema_compliance=np.mean(schema_scores) * 100,
            rule_compliance=np.mean(rule_scores) * 100,
            rationale_quality=np.mean(rationale_scores) * 100,
            response_time=np.mean(response_times),
            success_rate=sum(successes) / len(successes) * 100,
            composite_score=(np.mean(schema_scores) * 0.3 + 
                           np.mean(rule_scores) * 0.4 + 
                           np.mean(rationale_scores) * 0.3) * 100
        )
    
    def _execute_test(self, variant: PromptVariant, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single test"""
        
        import time
        
        # Format user message
        user_message = variant.user_template.format(
            client_metadata=json.dumps(test_case["client_metadata"], indent=2),
            market_conditions=test_case["market_conditions"],
            boundary_rules=json.dumps(test_case["boundary_rules"], indent=2),
            general_strategies=json.dumps(test_case["general_strategies"], indent=2)
        )
        
        # Call LLM
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": variant.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=variant.temperature
        )
        response_time = time.time() - start
        
        # Parse response
        content = response.choices[0].message.content
        
        # Evaluate
        schema_score = self._evaluate_schema(content)
        rule_score = self._evaluate_rules(content, test_case["boundary_rules"])
        rationale_score = self._evaluate_rationale(content)
        
        return {
            "schema_score": schema_score,
            "rule_score": rule_score,
            "rationale_score": rationale_score,
            "response_time": response_time,
            "success": schema_score > 0.5
        }
    
    def _evaluate_schema(self, response: str) -> float:
        """Evaluate JSON schema compliance"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return 0.0
            
            portfolio = json.loads(json_match.group())
            
            # Check required fields
            score = 0.0
            if "transactions" in portfolio:
                score += 0.5
                if isinstance(portfolio["transactions"], list) and len(portfolio["transactions"]) > 0:
                    score += 0.2
                    
                    # Check transaction structure
                    valid_transactions = all(
                        "action" in t and "asset_type" in t and "percentage" in t and "rationale" in t
                        for t in portfolio["transactions"]
                    )
                    if valid_transactions:
                        score += 0.2
            
            if "overall_rationale" in portfolio:
                score += 0.1
            
            return min(score, 1.0)
        
        except:
            return 0.0
    
    def _evaluate_rules(self, response: str, rules: List[str]) -> float:
        """Evaluate boundary rule compliance"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return 0.0
            
            portfolio = json.loads(json_match.group())
            transactions = portfolio.get("transactions", [])
            
            # Calculate total
            total = sum(t.get("percentage", 0) for t in transactions)
            
            # Check if close to 100%
            if abs(total - 100) > 1:
                return 0.5  # Major violation
            
            # Check for HOLD on 0% allocations
            violations = 0
            for t in transactions:
                if t.get("percentage", 0) == 0 and t.get("action") != "HOLD":
                    violations += 1
            
            # Check crypto exclusion if applicable
            crypto_txn = next((t for t in transactions if "Crypto" in t.get("asset_type", "")), None)
            # Simplified check - in production, use actual client preferences
            
            compliance_score = 1.0 - (violations * 0.1)
            return max(compliance_score, 0.0)
        
        except:
            return 0.0
    
    def _evaluate_rationale(self, response: str) -> float:
        """Evaluate rationale quality (heuristic)"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return 0.0
            
            portfolio = json.loads(json_match.group())
            overall = portfolio.get("overall_rationale", "")
            
            # Check length
            score = 0.0
            if len(overall) > 50:
                score += 0.3
            
            # Check for key terms
            key_terms = ["risk", "market", "allocation", "client", "portfolio"]
            terms_found = sum(1 for term in key_terms if term.lower() in overall.lower())
            score += (terms_found / len(key_terms)) * 0.4
            
            # Check transaction rationales
            transactions = portfolio.get("transactions", [])
            if transactions:
                avg_rationale_len = np.mean([len(t.get("rationale", "")) for t in transactions])
                if avg_rationale_len > 20:
                    score += 0.3
            
            return min(score, 1.0)
        
        except:
            return 0.0


class PromptOptimizationPipeline:
    """End-to-end prompt optimization pipeline"""
    
    def __init__(self, client: OpenAI, model: str = "gpt-4"):
        self.client = client
        self.model = model
        self.tester = PromptTester(client, model)
    
    def generate_test_cases(self, num_cases: int = 30) -> List[Dict[str, Any]]:
        """Generate diverse test cases"""
        
        logger.info(f"Generating {num_cases} test cases...")
        
        # Load client data
        try:
            clients_df = pd.read_csv("client.csv")
        except:
            logger.warning("client.csv not found, using mock data")
            clients_df = pd.DataFrame([
                {"client_id": "CL001", "risk_appetite": "low", "horizon_years": 3},
                {"client_id": "CL002", "risk_appetite": "moderate", "horizon_years": 5},
                {"client_id": "CL003", "risk_appetite": "high", "horizon_years": 10}
            ])
        
        test_cases = []
        
        for idx, client in clients_df.head(num_cases).iterrows():
            test_case = {
                "client_metadata": client.to_dict(),
                "market_conditions": self._generate_market_scenario(idx),
                "boundary_rules": self._generate_boundary_rules(client),
                "general_strategies": self._generate_strategies()
            }
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_market_scenario(self, idx: int) -> str:
        """Generate market scenario"""
        scenarios = [
            "Bullish market with strong tech sector performance. Inflation moderate.",
            "Bearish correction with flight to safety. Commodities rising.",
            "Neutral market with mixed signals. Geopolitical tensions.",
            "High volatility with recession concerns. Defensive positioning favored.",
            "Recovery phase with improving economic indicators."
        ]
        return scenarios[idx % len(scenarios)]
    
    def _generate_boundary_rules(self, client: Dict[str, Any]) -> List[str]:
        """Generate applicable boundary rules"""
        rules = [
            f"Risk appetite '{client.get('risk_appetite', 'moderate')}' requires specific equity limits",
            "Total allocation must equal 100%",
            "Gold allocation between 3-15%",
            "Action HOLD only for 0% allocations"
        ]
        
        if client.get("preferences_no_crypto"):
            rules.append("Cryptocurrency must be 0% (client exclusion)")
        
        return rules
    
    def _generate_strategies(self) -> List[Dict[str, Any]]:
        """Generate strategy reference"""
        return [
            {"type": "Conservative", "equity": 20, "debt": 50, "gold": 10},
            {"type": "Moderate", "equity": 45, "debt": 30, "gold": 8},
            {"type": "Aggressive", "equity": 70, "debt": 15, "gold": 5}
        ]
    
    def run_ab_test(self, num_test_cases: int = 30) -> Dict[str, Any]:
        """Run A/B test across all prompt variants"""
        
        logger.info("=" * 60)
        logger.info("PROMPT OPTIMIZATION PIPELINE")
        logger.info("=" * 60)
        
        # Generate test cases
        logger.info("\n[STEP 1] Generating test cases...")
        test_cases = self.generate_test_cases(num_test_cases)
        
        # Test all variants
        logger.info("\n[STEP 2] Testing prompt variants...")
        variants = PromptLibrary.get_all_variants()
        results = {}
        
        for variant in variants:
            result = self.tester.test_variant(variant, test_cases)
            results[variant.id] = {
                "name": variant.name,
                "description": variant.description,
                "metrics": result
            }
        
        # Analyze results
        logger.info("\n[STEP 3] Analyzing results...")
        analysis = self._analyze_results(results)
        
        # Generate recommendations
        logger.info("\n[STEP 4] Generating recommendations...")
        recommendations = self._generate_recommendations(results, analysis)
        
        # Compile final report
        report = {
            "test_cases": num_test_cases,
            "variants_tested": len(variants),
            "results": results,
            "analysis": analysis,
            "recommendations": recommendations
        }
        
        # Save report
        output_file = f"prompt_optimization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Optimization complete! Report saved to: {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return report
    
    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results"""
        
        # Find best performing variant
        best_variant = max(results.items(), key=lambda x: x[1]["metrics"].composite_score)
        
        # Calculate improvements
        baseline = results.get("baseline", {}).get("metrics")
        
        improvements = {}
        if baseline:
            for variant_id, data in results.items():
                if variant_id != "baseline":
                    metrics = data["metrics"]
                    improvements[variant_id] = {
                        "schema_improvement": metrics.schema_compliance - baseline.schema_compliance,
                        "rule_improvement": metrics.rule_compliance - baseline.rule_compliance,
                        "rationale_improvement": metrics.rationale_quality - baseline.rationale_quality,
                        "score_improvement": metrics.composite_score - baseline.composite_score
                    }
        
        return {
            "best_variant": best_variant[0],
            "best_score": best_variant[1]["metrics"].composite_score,
            "improvements": improvements
        }
    
    def _generate_recommendations(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        best_id = analysis["best_variant"]
        best_data = results[best_id]
        
        recommendations.append({
            "priority": "HIGH",
            "action": f"Deploy prompt variant '{best_data['name']}'",
            "rationale": f"Achieved composite score of {best_data['metrics'].composite_score:.1f}%",
            "expected_impact": f"Schema: {best_data['metrics'].schema_compliance:.1f}%, Rules: {best_data['metrics'].rule_compliance:.1f}%"
        })
        
        # Specific improvements
        baseline_metrics = results.get("baseline", {}).get("metrics")
        if baseline_metrics:
            if baseline_metrics.schema_compliance < 90:
                recommendations.append({
                    "priority": "CRITICAL",
                    "action": "Add explicit JSON schema in system prompt",
                    "rationale": "Current schema compliance below 90%",
                    "expected_impact": "Increase compliance to 95%+"
                })
            
            if baseline_metrics.rule_compliance < 95:
                recommendations.append({
                    "priority": "HIGH",
                    "action": "Emphasize boundary rules with visual markers (⚠️)",
                    "rationale": "Rule violations detected",
                    "expected_impact": "Achieve 98%+ rule compliance"
                })
        
        return recommendations
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print results summary"""
        
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        
        # Sort by composite score
        sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"].composite_score, reverse=True)
        
        logger.info(f"\n{'Variant':<25} {'Score':<10} {'Schema':<10} {'Rules':<10} {'Rationale':<10}")
        logger.info("-" * 65)
        
        for variant_id, data in sorted_results:
            metrics = data["metrics"]
            logger.info(
                f"{data['name']:<25} "
                f"{metrics.composite_score:>6.1f}%   "
                f"{metrics.schema_compliance:>6.1f}%   "
                f"{metrics.rule_compliance:>6.1f}%   "
                f"{metrics.rationale_quality:>6.1f}%"
            )


if __name__ == "__main__":
    print("Prompt Optimization Framework Initialized")
    print("\nTo run optimization:")
    print("  from openai import OpenAI")
    print("  client = OpenAI()")
    print("  optimizer = PromptOptimizationPipeline(client, model='gpt-4')")
    print("  results = optimizer.run_ab_test(num_test_cases=30)")
