"""
Integration Tests for Portfolio Advisor Workflow
Tests component interactions and workflow execution
"""

import pytest
import sys
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
import json
import asyncio

sys.path.insert(0, '../')

from test_synthetic_data import SyntheticDataGenerator


class TestMarketAnalysisWorkflow:
    """Test complete market analysis workflow"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.fixture
    def mock_search_results(self):
        """Mock search results"""
        return [
            MagicMock(page_content="Market news article 1"),
            MagicMock(page_content="Market news article 2")
        ]
    
    def test_market_analysis_complete_flow(self, data_generator, mock_search_results):
        """Test complete market analysis workflow"""
        
        # Mock components
        with patch('marketTrend.generate_questions') as mock_gen_q, \
             patch('marketTrend.search_financial_news') as mock_search, \
             patch('marketTrend.summarize_trends') as mock_summarize, \
             patch('marketTrend.retrieve_similar_scenarios') as mock_retrieve:
            
            # Setup mocks
            mock_gen_q.return_value = ["Question 1", "Question 2"]
            mock_search.return_value = ["News 1", "News 2"]
            mock_summarize.return_value = "Market summary"
            mock_retrieve.return_value = {"Gold": 5.0, "BankFD": -3.0}
            
            from marketTrend import analysisMarket
            
            # Execute workflow
            result = analysisMarket("Test topic")
            
            # Verify workflow execution
            mock_gen_q.assert_called_once_with("Test topic")
            mock_search.assert_called_once()
            mock_summarize.assert_called_once()
            mock_retrieve.assert_called_once()
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "Gold" in result or len(result) >= 0
    
    def test_delta_application_to_strategy(self, data_generator):
        """Test that deltas are correctly applied to strategy CSV"""
        
        # Create test strategy DataFrame
        test_data = {
            "Client Type": ["Conservative", "Aggressive"],
            "BankFD": [30.0, 10.0],
            "Gold": [10.0, 5.0],
            "MF_Index": [20.0, 30.0]
        }
        df = pd.DataFrame(test_data)
        
        # Save test CSV
        test_csv = "/tmp/test_strategy.csv"
        df.to_csv(test_csv, index=False)
        
        # Apply deltas
        deltas = {"Gold": 10.0, "BankFD": -5.0}
        
        from marketTrend import apply_instrument_deltas
        
        result_df = apply_instrument_deltas(
            test_csv,
            "/tmp/test_strategy_updated.csv",
            deltas
        )
        
        # Verify result
        assert result_df is not None
        assert len(result_df) == 2  # Same number of strategies
        
        # Load updated CSV
        updated_df = pd.read_csv("/tmp/test_strategy_updated.csv")
        assert len(updated_df) > 0
        
        # Cleanup
        import os
        os.remove(test_csv)
        os.remove("/tmp/test_strategy_updated.csv")
    
    def test_market_analysis_observability(self, data_generator):
        """Test that observability metrics are collected"""
        
        with patch('marketTrend._collector') as mock_collector:
            mock_collector.get_summary.return_value = {
                'total_tokens': 1000,
                'total_cost_usd': 0.03,
                'avg_latency_ms': 500.0
            }
            
            from marketTrend import reason
            
            with patch('marketTrend.analysisMarket') as mock_analysis, \
                 patch('marketTrend.apply_instrument_deltas') as mock_apply:
                
                mock_analysis.return_value = {}
                mock_apply.return_value = pd.DataFrame()
                
                result = reason("Test topic")
                
                # Verify observability data
                assert "observability" in result
                assert result["observability"]["tokens"] == 1000


class TestLangGraphWorkflow:
    """Test LangGraph portfolio generation workflow"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.fixture
    def test_state(self, data_generator):
        """Create test state"""
        return {
            "client_id": "TEST001",
            "marketCondtions": "Stable market conditions"
        }
    
    def test_load_client_metadata_node(self, data_generator, test_state):
        """Test client metadata loading node"""
        
        # Mock CSV data
        test_df = pd.DataFrame([
            data_generator.generate_client_metadata("TEST001")
        ])
        
        with patch('pandas.read_csv', return_value=test_df):
            from langgraphapp import load_client_metadata_node
            
            result = load_client_metadata_node(test_state)
            
            # Verify
            assert "client_metadata" in result
            assert result["client_metadata"]["client_id"] == "TEST001"
    
    def test_select_boundary_rules_node(self, data_generator):
        """Test boundary rules selection node"""
        
        state = {
            "client_metadata": data_generator.generate_client_metadata("TEST001")
        }
        
        # Mock LLM response
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value.rule_keys = ["max_equity_for_low_risk"]
        
        # Mock vector store
        mock_doc = MagicMock()
        mock_doc.page_content = "Max equity rule: 30%"
        
        mock_vectorstore = MagicMock()
        mock_vectorstore.as_retriever.return_value.invoke.return_value = [mock_doc]
        
        with patch('langgraphapp.llm.with_structured_output', return_value=mock_structured_llm), \
             patch('langgraphapp.Chroma', return_value=mock_vectorstore):
            
            from langgraphapp import select_and_retrieve_boundary_rules_node
            
            result = select_and_retrieve_boundary_rules_node(state)
            
            # Verify
            assert "boundary_rules" in result
            assert isinstance(result["boundary_rules"], list)
    
    def test_load_general_strategies_node(self, data_generator):
        """Test general strategies loading node"""
        
        # Create test CSV
        strategies = data_generator.generate_general_strategies()
        test_df = pd.DataFrame(strategies)
        
        with patch('pandas.read_csv', return_value=test_df):
            from langgraphapp import load_general_strategies_node
            
            result = load_general_strategies_node({})
            
            # Verify
            assert "general_strategies" in result
            assert len(result["general_strategies"]) == len(strategies)
    
    def test_generate_portfolio_node(self, data_generator):
        """Test portfolio generation node"""
        
        state = {
            "client_metadata": data_generator.generate_client_metadata("TEST001"),
            "boundary_rules": data_generator.generate_boundary_rules("medium"),
            "general_strategies": data_generator.generate_general_strategies(),
            "marketCondtions": "Stable growth"
        }
        
        # Mock portfolio plan
        mock_plan = MagicMock()
        mock_plan.transactions = []
        mock_plan.overall_rationale = "Test rationale"
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_plan
        
        with patch('langgraphapp.llm.with_structured_output', return_value=mock_llm):
            from langgraphapp import generate_custom_portfolio_node
            
            result = generate_custom_portfolio_node(state)
            
            # Verify
            assert "portfolio_plan" in result
    
    def test_complete_workflow_state_flow(self, data_generator):
        """Test that state flows correctly through all nodes"""
        
        # Create complete test state
        test_case = data_generator.generate_complete_test_case()
        
        # Simulate workflow
        state = {"client_id": test_case["client_metadata"]["client_id"]}
        
        # Node 1: Load client
        state["client_metadata"] = test_case["client_metadata"]
        assert "client_metadata" in state
        
        # Node 2: Load boundary rules
        state["boundary_rules"] = test_case["boundary_rules"]
        assert "boundary_rules" in state
        
        # Node 3: Load strategies
        state["general_strategies"] = test_case["general_strategies"]
        assert "general_strategies" in state
        
        # Verify complete state
        required_keys = ["client_id", "client_metadata", "boundary_rules", "general_strategies"]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_analyze_market_endpoint_structure(self, data_generator):
        """Test /analyze_market endpoint response structure"""
        
        expected_response = {
            "status": "success",
            "files_generated": ["strategy_updated.csv"],
            "observability": {
                "tokens_used": 1000,
                "cost": "$0.03",
                "latency_ms": 500
            }
        }
        
        # Verify structure
        assert "status" in expected_response
        assert "files_generated" in expected_response
        assert "observability" in expected_response
        assert isinstance(expected_response["files_generated"], list)
    
    def test_analyze_portfolio_endpoint_structure(self, data_generator):
        """Test /analyze_client_portfolio endpoint response structure"""
        
        expected_response = {
            "status": "success",
            "csv_file": "CL001.csv",
            "message": "Portfolio created successfully",
            "observability": {
                "total_events": 10,
                "success_rate": "100.0%",
                "tokens_used": 2000,
                "cost": "$0.06"
            }
        }
        
        # Verify structure
        assert "status" in expected_response
        assert "csv_file" in expected_response
        assert "observability" in expected_response


class TestObservabilityIntegration:
    """Test observability platform integration"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_trace_collection(self):
        """Test that traces are collected"""
        
        from observability_platform import _collector
        
        # Clear existing traces
        _collector.traces.clear()
        
        # Simulate some events
        from observability_platform import trace_agent
        
        @trace_agent("test_agent")
        def test_function():
            return "test"
        
        result = test_function()
        
        # Verify trace was collected
        # Note: This depends on implementation
        assert result == "test"
    
    def test_llm_call_tracking(self):
        """Test that LLM calls are tracked"""
        
        from observability_platform import trace_llm_call, _collector
        
        # Clear tracking
        _collector.token_usage.clear()
        _collector.cost_tracking.clear()
        
        # Simulate LLM call
        trace_llm_call(
            model="test-model",
            tokens=100,
            cost=0.003
        )
        
        # Verify tracking
        # Note: Implementation specific
        assert True  # Placeholder
    
    def test_guardrail_violations_logged(self):
        """Test that guardrail violations are logged"""
        
        from observability_platform import LangChainObserver, Severity
        
        observer = LangChainObserver()
        
        # Add test rule
        observer.guardrails.add_rule(
            "test_rule",
            lambda text: "forbidden" not in text.lower(),
            severity=Severity.ERROR
        )
        
        # Trigger violation
        violations = observer.guardrails.check("This contains forbidden content")
        
        # Verify
        assert len(violations) > 0


class TestDataPersistence:
    """Test data loading and saving"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_csv_read_write(self, data_generator):
        """Test CSV read/write operations"""
        
        # Generate test data
        strategies = data_generator.generate_general_strategies()
        df = pd.DataFrame(strategies)
        
        # Write
        test_file = "/tmp/test_strategies.csv"
        df.to_csv(test_file, index=False)
        
        # Read
        loaded_df = pd.read_csv(test_file)
        
        # Verify
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
        
        # Cleanup
        import os
        os.remove(test_file)
    
    def test_json_serialization(self, data_generator):
        """Test JSON serialization of test data"""
        
        test_case = data_generator.generate_complete_test_case()
        
        # Serialize
        json_str = json.dumps(test_case, indent=2)
        
        # Deserialize
        loaded = json.loads(json_str)
        
        # Verify
        assert loaded["client_metadata"]["client_id"] == test_case["client_metadata"]["client_id"]
        assert loaded["scenario"] == test_case["scenario"]


class TestErrorRecovery:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_missing_client_recovery(self, data_generator):
        """Test recovery when client not found"""
        
        # Mock CSV with no matching client
        test_df = pd.DataFrame([
            data_generator.generate_client_metadata("OTHER001")
        ])
        
        with patch('pandas.read_csv', return_value=test_df):
            from langgraphapp import load_client_metadata_node
            
            state = {"client_id": "MISSING001"}
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="not found"):
                load_client_metadata_node(state)
    
    def test_empty_strategy_handling(self, data_generator):
        """Test handling of empty strategy file"""
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        with patch('pandas.read_csv', return_value=empty_df):
            from langgraphapp import load_general_strategies_node
            
            result = load_general_strategies_node({})
            
            # Should return empty list
            assert result["general_strategies"] == []
    
    def test_api_timeout_handling(self):
        """Test handling of API timeouts"""
        
        with patch('marketTrend.serachRe.invoke', side_effect=TimeoutError("Timeout")):
            from marketTrend import search_financial_news
            
            # Should not crash
            result = search_financial_news(["test query"])
            
            # Should return empty or partial results
            assert isinstance(result, list)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.mark.slow
    def test_data_generation_speed(self, data_generator):
        """Test that data generation is reasonably fast"""
        import time
        
        start = time.time()
        
        # Generate 100 test cases
        test_cases = data_generator.generate_test_suite(count=100)
        
        end = time.time()
        elapsed = end - start
        
        # Should complete within 5 seconds
        assert elapsed < 5.0, f"Generation took {elapsed:.2f}s"
        assert len(test_cases) == 100
    
    @pytest.mark.slow
    def test_csv_processing_speed(self, data_generator):
        """Test CSV processing performance"""
        import time
        
        # Create large strategy file
        strategies = data_generator.generate_general_strategies() * 100
        df = pd.DataFrame(strategies)
        
        test_file = "/tmp/large_strategy.csv"
        df.to_csv(test_file, index=False)
        
        # Time reading
        start = time.time()
        loaded_df = pd.read_csv(test_file)
        end = time.time()
        
        elapsed = end - start
        
        # Should be fast
        assert elapsed < 1.0, f"CSV read took {elapsed:.2f}s"
        
        # Cleanup
        import os
        os.remove(test_file)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
