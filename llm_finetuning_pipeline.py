"""
End-to-End LLM Fine-tuning Pipeline for Portfolio Advisor
Supports: GPT-3.5/4, Azure OpenAI, and Local Models
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import openai
from openai import OpenAI, AzureOpenAI
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    LOCAL = "local"


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    provider: ModelProvider
    base_model: str  # e.g., "gpt-3.5-turbo-1106" or "gpt-4-0613"
    training_file: str
    validation_file: Optional[str] = None
    n_epochs: int = 3
    batch_size: int = 1
    learning_rate_multiplier: float = 1.0
    suffix: str = "portfolio-advisor"
    
    # Azure specific
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_deployment: Optional[str] = None
    
    # OpenAI specific
    openai_api_key: Optional[str] = None


class TrainingDataGenerator:
    """Generate training data from existing system logs and expert examples"""
    
    def __init__(self, output_dir: str = "fine_tuning_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_training_example(
        self,
        client_metadata: Dict[str, Any],
        market_conditions: str,
        boundary_rules: List[str],
        general_strategies: List[Dict[str, Any]],
        expert_portfolio: Dict[str, Any],
        expert_rationale: str
    ) -> Dict[str, Any]:
        """Create a single training example in OpenAI format"""
        
        # System message with role definition
        system_message = """You are an expert financial portfolio advisor. Your task is to create personalized investment portfolios based on:
1. Client profile (risk appetite, goals, horizon, preferences)
2. Current market conditions
3. Regulatory boundary rules (must be strictly followed)
4. General strategy guidelines (directional reference)

You must output a valid JSON with:
- transactions: List of asset allocations with action (BUY/SELL/HOLD), percentage, and rationale
- overall_rationale: Summary of key decision factors

CRITICAL RULES:
- Total allocations must sum to 100%
- All boundary rules must be satisfied
- If client excludes crypto (preferences_no_crypto=true), cryptocurrency must be 0%
- Action HOLD only for 0% allocations or unchanged positions
- Each rationale must reference specific client needs and market factors"""

        # User message with all context
        user_message = f"""Create a portfolio for this client:

CLIENT PROFILE:
{json.dumps(client_metadata, indent=2)}

MARKET CONDITIONS:
{market_conditions}

BOUNDARY RULES (MUST COMPLY):
{json.dumps(boundary_rules, indent=2)}

GENERAL STRATEGIES (REFERENCE):
{json.dumps(general_strategies, indent=2)}

Provide a complete portfolio plan as JSON."""

        # Assistant message with expert answer
        assistant_message = json.dumps({
            "transactions": expert_portfolio["transactions"],
            "overall_rationale": expert_rationale
        }, indent=2)
        
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
    
    def generate_from_csv(
        self,
        client_csv: str = "client.csv",
        portfolio_results_csv: str = "expert_portfolios.csv",
        num_samples: int = 500
    ) -> str:
        """Generate training data from CSV files"""
        
        logger.info(f"Generating {num_samples} training examples...")
        
        # Load client data
        clients = pd.read_csv(client_csv)
        
        # Mock market conditions (in production, load from actual data)
        market_scenarios = [
            "Bullish market with inflation concerns. Tech sector outperforming. Interest rates stable.",
            "Bearish correction phase. Flight to safety assets. Commodity prices rising.",
            "Neutral market. Mixed sector performance. Geopolitical tensions moderate.",
            "High volatility. Recession fears. Central banks tightening policy.",
            "Recovery phase. Strong economic indicators. Energy sector leading."
        ]
        
        # Mock boundary rules
        boundary_rules_pool = [
            "If risk_appetite='low', total equity must not exceed 30%",
            "If risk_appetite='moderate', total equity range 30-60%",
            "If risk_appetite='high', total equity can be 60-80%",
            "Gold allocation must be between 3-15% for all clients",
            "Cryptocurrency capped at 10% for high-risk, 0% for others",
            "If horizon_years < 3, equity allocation reduced by 20%",
            "If liquidity_need='high', BankFD + DebtBond >= 40%",
            "Real estate allocation: 5-15% for clients with horizon > 5 years"
        ]
        
        training_examples = []
        
        for idx, client in clients.head(num_samples).iterrows():
            # Create client metadata
            client_metadata = client.to_dict()
            
            # Select market condition
            market_condition = market_scenarios[idx % len(market_scenarios)]
            
            # Select relevant boundary rules
            boundary_rules = boundary_rules_pool[:5]  # Use first 5 rules
            
            # Mock strategy (simplified)
            strategies = [
                {"Client Type": "Conservative", "Equity": 20, "Debt": 50, "Gold": 10},
                {"Client Type": "Moderate", "Equity": 45, "Debt": 30, "Gold": 8}
            ]
            
            # Generate expert portfolio (mock - in production use actual expert data)
            expert_portfolio = self._generate_expert_portfolio(client_metadata)
            expert_rationale = self._generate_expert_rationale(client_metadata, market_condition)
            
            example = self.create_training_example(
                client_metadata=client_metadata,
                market_conditions=market_condition,
                boundary_rules=boundary_rules,
                general_strategies=strategies,
                expert_portfolio=expert_portfolio,
                expert_rationale=expert_rationale
            )
            
            training_examples.append(example)
        
        # Save to JSONL
        output_file = self.output_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(output_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Saved {len(training_examples)} examples to {output_file}")
        return str(output_file)
    
    def _generate_expert_portfolio(self, client: Dict[str, Any]) -> Dict[str, Any]:
        """Mock expert portfolio generation (replace with actual expert data)"""
        
        risk = client.get('risk_appetite', 'moderate')
        
        if risk == 'low':
            return {
                "transactions": [
                    {"action": "BUY", "asset_type": "BankFD", "percentage": 25, "rationale": "Capital preservation priority"},
                    {"action": "BUY", "asset_type": "DebtBond", "percentage": 25, "rationale": "Stable income generation"},
                    {"action": "BUY", "asset_type": "MF_Index", "percentage": 15, "rationale": "Diversified equity exposure"},
                    {"action": "BUY", "asset_type": "MF_Flexi", "percentage": 10, "rationale": "Balanced growth"},
                    {"action": "BUY", "asset_type": "Gold", "percentage": 10, "rationale": "Inflation hedge"},
                    {"action": "BUY", "asset_type": "RealEstate", "percentage": 10, "rationale": "Long-term stability"},
                    {"action": "HOLD", "asset_type": "Cryptocurrency", "percentage": 0, "rationale": "Risk profile exclusion"},
                    {"action": "BUY", "asset_type": "EQ_Banking", "percentage": 2, "rationale": "Defensive sector"},
                    {"action": "BUY", "asset_type": "EQ_FMCG", "percentage": 2, "rationale": "Stable demand"},
                    {"action": "BUY", "asset_type": "Silver", "percentage": 1, "rationale": "Minor commodity exposure"}
                ]
            }
        else:
            return {
                "transactions": [
                    {"action": "BUY", "asset_type": "BankFD", "percentage": 10, "rationale": "Liquidity buffer"},
                    {"action": "BUY", "asset_type": "DebtBond", "percentage": 15, "rationale": "Income stability"},
                    {"action": "BUY", "asset_type": "MF_Index", "percentage": 20, "rationale": "Core equity position"},
                    {"action": "BUY", "asset_type": "MF_SmallCap", "percentage": 15, "rationale": "Growth potential"},
                    {"action": "BUY", "asset_type": "EQ_IT", "percentage": 10, "rationale": "Sector leadership"},
                    {"action": "BUY", "asset_type": "EQ_Banking", "percentage": 8, "rationale": "Economic cycle play"},
                    {"action": "BUY", "asset_type": "Gold", "percentage": 8, "rationale": "Portfolio hedge"},
                    {"action": "BUY", "asset_type": "RealEstate", "percentage": 10, "rationale": "Real asset allocation"},
                    {"action": "BUY", "asset_type": "Cryptocurrency", "percentage": 4, "rationale": "High-growth allocation"},
                ]
            }
    
    def _generate_expert_rationale(self, client: Dict[str, Any], market: str) -> str:
        """Generate expert overall rationale"""
        risk = client.get('risk_appetite', 'moderate')
        horizon = client.get('horizon_years', 5)
        
        return f"Portfolio designed for {risk} risk profile with {horizon}-year horizon. " \
               f"Allocation balances {market.lower()} considering client liquidity needs and " \
               f"regulatory constraints. Emphasis on capital preservation with measured growth exposure."
    
    def validate_training_data(self, jsonl_file: str) -> Dict[str, Any]:
        """Validate training data format and quality"""
        
        logger.info(f"Validating {jsonl_file}...")
        
        issues = []
        examples = []
        
        with open(jsonl_file, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    example = json.loads(line)
                    examples.append(example)
                    
                    # Check structure
                    if "messages" not in example:
                        issues.append(f"Line {idx}: Missing 'messages' key")
                        continue
                    
                    messages = example["messages"]
                    if len(messages) != 3:
                        issues.append(f"Line {idx}: Expected 3 messages (system, user, assistant)")
                    
                    # Check roles
                    roles = [m["role"] for m in messages]
                    if roles != ["system", "user", "assistant"]:
                        issues.append(f"Line {idx}: Incorrect role sequence")
                    
                    # Validate assistant response is valid JSON
                    assistant_content = messages[2]["content"]
                    try:
                        portfolio = json.loads(assistant_content)
                        if "transactions" not in portfolio or "overall_rationale" not in portfolio:
                            issues.append(f"Line {idx}: Assistant response missing required keys")
                    except json.JSONDecodeError:
                        issues.append(f"Line {idx}: Assistant response is not valid JSON")
                
                except json.JSONDecodeError:
                    issues.append(f"Line {idx}: Invalid JSON")
        
        validation_result = {
            "total_examples": len(examples),
            "issues_found": len(issues),
            "issues": issues[:10],  # First 10 issues
            "status": "PASS" if len(issues) == 0 else "FAIL"
        }
        
        logger.info(f"Validation: {validation_result['status']} - {len(examples)} examples, {len(issues)} issues")
        
        return validation_result


class FineTuner:
    """Manage fine-tuning jobs"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        
        if config.provider == ModelProvider.OPENAI:
            self.client = OpenAI(api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"))
        elif config.provider == ModelProvider.AZURE:
            self.client = AzureOpenAI(
                api_key=config.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            )
    
    def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI"""
        
        logger.info(f"Uploading training file: {file_path}")
        
        with open(file_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        logger.info(f"Uploaded file ID: {file_id}")
        
        return file_id
    
    def create_fine_tuning_job(self, training_file_id: str, validation_file_id: Optional[str] = None) -> str:
        """Create fine-tuning job"""
        
        logger.info(f"Creating fine-tuning job for model: {self.config.base_model}")
        
        job_params = {
            "training_file": training_file_id,
            "model": self.config.base_model,
            "suffix": self.config.suffix,
            "hyperparameters": {
                "n_epochs": self.config.n_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate_multiplier": self.config.learning_rate_multiplier
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
        
        response = self.client.fine_tuning.jobs.create(**job_params)
        
        job_id = response.id
        logger.info(f"Fine-tuning job created: {job_id}")
        
        return job_id
    
    def monitor_job(self, job_id: str, poll_interval: int = 60):
        """Monitor fine-tuning job progress"""
        
        logger.info(f"Monitoring job: {job_id}")
        
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            logger.info(f"Status: {status}")
            
            if status == "succeeded":
                logger.info(f"Fine-tuning completed! Model: {job.fine_tuned_model}")
                return job.fine_tuned_model
            
            elif status == "failed":
                logger.error(f"Fine-tuning failed: {job.error}")
                raise Exception(f"Fine-tuning failed: {job.error}")
            
            elif status == "cancelled":
                logger.warning("Fine-tuning was cancelled")
                return None
            
            else:
                # Still running
                import time
                time.sleep(poll_interval)
    
    def list_jobs(self, limit: int = 10):
        """List recent fine-tuning jobs"""
        
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        
        jobs_info = []
        for job in jobs.data:
            jobs_info.append({
                "id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "created_at": job.created_at
            })
        
        return jobs_info
    
    def cancel_job(self, job_id: str):
        """Cancel a running fine-tuning job"""
        
        logger.info(f"Cancelling job: {job_id}")
        self.client.fine_tuning.jobs.cancel(job_id)


class ModelEvaluator:
    """Evaluate fine-tuned models"""
    
    def __init__(self, base_model: str, fine_tuned_model: str, client: OpenAI):
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model
        self.client = client
    
    def load_test_data(self, test_file: str) -> List[Dict[str, Any]]:
        """Load test dataset"""
        
        test_examples = []
        with open(test_file, 'r') as f:
            for line in f:
                test_examples.append(json.loads(line))
        
        return test_examples
    
    def evaluate_model(self, model_name: str, test_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model on test set"""
        
        logger.info(f"Evaluating model: {model_name}")
        
        results = {
            "model": model_name,
            "total_tests": len(test_examples),
            "successful": 0,
            "failed": 0,
            "schema_errors": 0,
            "rule_violations": 0,
            "avg_response_time": 0
        }
        
        response_times = []
        
        for idx, example in enumerate(test_examples):
            try:
                messages = example["messages"][:2]  # System + User
                
                import time
                start = time.time()
                
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3
                )
                
                response_time = time.time() - start
                response_times.append(response_time)
                
                # Parse response
                assistant_response = response.choices[0].message.content
                
                try:
                    portfolio = json.loads(assistant_response)
                    
                    # Validate schema
                    if "transactions" in portfolio and "overall_rationale" in portfolio:
                        results["successful"] += 1
                        
                        # Check basic rules
                        total_pct = sum(t.get("percentage", 0) for t in portfolio["transactions"])
                        if abs(total_pct - 100) > 0.1:
                            results["rule_violations"] += 1
                    else:
                        results["schema_errors"] += 1
                
                except json.JSONDecodeError:
                    results["schema_errors"] += 1
            
            except Exception as e:
                logger.error(f"Test {idx} failed: {e}")
                results["failed"] += 1
        
        results["avg_response_time"] = sum(response_times) / len(response_times) if response_times else 0
        results["success_rate"] = results["successful"] / results["total_tests"] * 100
        
        return results
    
    def compare_models(self, test_file: str) -> Dict[str, Any]:
        """Compare base model vs fine-tuned model"""
        
        test_examples = self.load_test_data(test_file)
        
        # Sample subset for faster evaluation
        test_sample = test_examples[:50]
        
        base_results = self.evaluate_model(self.base_model, test_sample)
        ft_results = self.evaluate_model(self.fine_tuned_model, test_sample)
        
        comparison = {
            "base_model": base_results,
            "fine_tuned_model": ft_results,
            "improvements": {
                "success_rate": ft_results["success_rate"] - base_results["success_rate"],
                "schema_errors": base_results["schema_errors"] - ft_results["schema_errors"],
                "rule_violations": base_results["rule_violations"] - ft_results["rule_violations"],
                "response_time": base_results["avg_response_time"] - ft_results["avg_response_time"]
            }
        }
        
        logger.info(f"\nComparison Results:")
        logger.info(f"Success Rate Improvement: {comparison['improvements']['success_rate']:.2f}%")
        logger.info(f"Schema Errors Reduced: {comparison['improvements']['schema_errors']}")
        logger.info(f"Rule Violations Reduced: {comparison['improvements']['rule_violations']}")
        
        return comparison


# ============================================
# MAIN EXECUTION PIPELINE
# ============================================

async def run_full_pipeline():
    """Execute end-to-end fine-tuning pipeline"""
    
    logger.info("=" * 60)
    logger.info("STARTING END-TO-END LLM FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Generate Training Data
    logger.info("\n[STEP 1] Generating Training Data...")
    data_generator = TrainingDataGenerator()
    training_file = data_generator.generate_from_csv(num_samples=500)
    
    # Step 2: Validate Training Data
    logger.info("\n[STEP 2] Validating Training Data...")
    validation_result = data_generator.validate_training_data(training_file)
    
    if validation_result["status"] == "FAIL":
        logger.error("Training data validation failed!")
        logger.error(f"Issues: {validation_result['issues']}")
        return
    
    # Step 3: Split into train/validation
    logger.info("\n[STEP 3] Splitting into train/validation sets...")
    with open(training_file, 'r') as f:
        all_examples = [json.loads(line) for line in f]
    
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    train_file = training_file.replace('.jsonl', '_train.jsonl')
    val_file = training_file.replace('.jsonl', '_val.jsonl')
    
    with open(train_file, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')
    
    with open(val_file, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')
    
    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Validation examples: {len(val_examples)}")
    
    # Step 4: Configure Fine-tuning
    logger.info("\n[STEP 4] Configuring Fine-tuning Job...")
    config = FineTuningConfig(
        provider=ModelProvider.OPENAI,
        base_model="gpt-3.5-turbo-1106",
        training_file=train_file,
        validation_file=val_file,
        n_epochs=3,
        batch_size=1,
        learning_rate_multiplier=1.0,
        suffix="portfolio-advisor-v1"
    )
    
    # Step 5: Upload Files and Start Fine-tuning
    logger.info("\n[STEP 5] Starting Fine-tuning...")
    fine_tuner = FineTuner(config)
    
    train_file_id = fine_tuner.upload_training_file(train_file)
    val_file_id = fine_tuner.upload_training_file(val_file)
    
    job_id = fine_tuner.create_fine_tuning_job(train_file_id, val_file_id)
    
    # Step 6: Monitor Job
    logger.info("\n[STEP 6] Monitoring Fine-tuning Job...")
    fine_tuned_model = fine_tuner.monitor_job(job_id, poll_interval=60)
    
    if not fine_tuned_model:
        logger.error("Fine-tuning did not complete successfully")
        return
    
    # Step 7: Evaluate Model
    logger.info("\n[STEP 7] Evaluating Fine-tuned Model...")
    evaluator = ModelEvaluator(
        base_model=config.base_model,
        fine_tuned_model=fine_tuned_model,
        client=fine_tuner.client
    )
    
    comparison = evaluator.compare_models(val_file)
    
    # Step 8: Save Results
    logger.info("\n[STEP 8] Saving Results...")
    results_file = f"fine_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    final_results = {
        "job_id": job_id,
        "base_model": config.base_model,
        "fine_tuned_model": fine_tuned_model,
        "training_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "comparison": comparison,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        json.dumps(final_results, f, indent=2)
    
    logger.info(f"\n✓ Pipeline completed successfully!")
    logger.info(f"✓ Fine-tuned model: {fine_tuned_model}")
    logger.info(f"✓ Results saved to: {results_file}")
    
    return final_results


if __name__ == "__main__":
    """
    Usage:
    1. Set environment variables: OPENAI_API_KEY or AZURE_OPENAI_API_KEY
    2. Ensure client.csv exists with client data
    3. Run: python llm_finetuning_pipeline.py
    """
    
    # Quick test without full pipeline
    print("LLM Fine-tuning Pipeline Initialized")
    print("\nTo run full pipeline:")
    print("  import asyncio")
    print("  asyncio.run(run_full_pipeline())")
    
    # Generate sample data only
    data_gen = TrainingDataGenerator()
    training_file = data_gen.generate_from_csv(num_samples=10)
    validation = data_gen.validate_training_data(training_file)
    print(f"\nGenerated sample data: {training_file}")
    print(f"Validation: {validation['status']}")
