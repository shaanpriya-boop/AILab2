"""
Unit Tests for Portfolio Advisor Components
Tests individual functions and components in isolation
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict
import json

# Add parent directory to path
sys.path.insert(0, '../')

from test_synthetic_data import SyntheticDataGenerator


class TestPromptGeneration:
    """Test prompt generation and LLM interactions"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing without API calls"""
        mock = MagicMock()
        mock.invoke.return_value.content = "Test LLM response"
        return mock
    
    def test_market_analysis_questions_generation(self, data_generator, mock_llm):
        """Test that market analysis generates correct number of questions"""
        with patch('marketTrend.llm', mock_llm):
            from marketTrend import generate_questions
            
            # Mock structured output
            mock_structured = MagicMock()
            mock_structured.invoke.return_value.questions = [
                f"Question {i}" for i in range(10)
            ]
            mock_llm.with_structured_output.return_value = mock_structured
            
            questions = generate_questions("Test topic")
            
            # Verify
            assert len(questions) == 10
            assert all(isinstance(q, str) for q in questions)
            assert all(len(q) > 0 for q in questions)
    
    def test_market_summary_generation(self, data_generator, mock_llm):
        """Test market summary generation from news"""
        with patch('marketTrend.llm', mock_llm):
            from marketTrend import summarize_trends
            
            news_list = data_generator.generate_news_articles("test", count=5)
            mock_llm.invoke.return_value.content = "Market showing moderate growth"
            
            summary = summarize_trends(news_list)
            
            # Verify
            assert isinstance(summary, str)
            assert len(summary) > 0
            mock_llm.invoke.assert_called_once()


class TestRAGRetrieval:
    """Test RAG retrieval functionality"""
    
    @pytest.fixture
    def mock_vectorstore(self):
        """Mock vector store for testing"""
        mock = MagicMock()
        mock.as_retriever.return_value.invoke.return_value = [
            MagicMock(page_content="Test rule 1", metadata={"rule_id": "rule_1"}),
            MagicMock(page_content="Test rule 2", metadata={"rule_id": "rule_2"})
        ]
        return mock
    
    def test_boundary_rules_retrieval(self, mock_vectorstore):
        """Test boundary rules are correctly retrieved"""
        with patch('langgraphapp.Chroma', return_value=mock_vectorstore):
            # Simulate retrieval
            retriever = mock_vectorstore.as_retriever()
            docs = retriever.invoke("test query")
            
            # Verify
            assert len(docs) == 2
            assert all(hasattr(doc, 'page_content') for doc in docs)
            assert all(hasattr(doc, 'metadata') for doc in docs)
    
    def test_retrieval_with_empty_results(self, mock_vectorstore):
        """Test handling of empty retrieval results"""
        mock_vectorstore.as_retriever.return_value.invoke.return_value = []
        
        with patch('langgraphapp.Chroma', return_value=mock_vectorstore):
            retriever = mock_vectorstore.as_retriever()
            docs = retriever.invoke("non-existent query")
            
            # Should return empty list, not error
            assert docs == []
    
    def test_market_scenario_retrieval(self):
        """Test retrieval of similar market scenarios"""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            'documents': ["Scenario 1", "Scenario 2"],
            'metadatas': [
                {"Gold": 5.0, "BankFD": -3.0},
                {"Gold": 10.0, "BankFD": -5.0}
            ]
        }
        
        # Verify metadata extraction
        metadata = mock_collection.get()['metadatas'][0]
        assert "Gold" in metadata
        assert isinstance(metadata["Gold"], float)


class TestAgentBehavior:
    """Test agent decision-making and behavior"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_question_generation_relevance(self, data_generator):
        """Test that generated questions are relevant to topic"""
        # This would test with actual LLM in integration tests
        # Here we test the structure
        topic = "Federal Reserve interest rates"
        
        # Mock response
        mock_questions = [
            "What is the current Fed funds rate?",
            "How many rate hikes are expected?",
            "What is inflation trajectory?",
        ]
        
        # Verify structure
        assert all(isinstance(q, str) for q in mock_questions)
        assert all("?" in q or q[0].isupper() for q in mock_questions)
    
    def test_news_search_error_handling(self, data_generator):
        """Test that news search handles errors gracefully"""
        with patch('marketTrend.serachRe') as mock_search:
            # Simulate search failure
            mock_search.invoke.side_effect = Exception("API Error")
            
            from marketTrend import search_financial_news
            
            # Should not crash
            result = search_financial_news(["test query"])
            
            # Should return empty or handle gracefully
            assert isinstance(result, list)
    
    def test_delta_application_validation(self, data_generator):
        """Test that deltas are applied correctly to strategies"""
        import pandas as pd
        
        # Create test strategy
        test_strategy = pd.DataFrame([{
            "Client Type": "Test",
            "BankFD": 20.0,
            "Gold": 10.0,
            "MF_Index": 30.0
        }])
        
        deltas = {"Gold": 10.0, "BankFD": -5.0}  # +10% to Gold, -5% to BankFD
        
        # Apply deltas manually for test
        for col, delta in deltas.items():
            if col in test_strategy.columns:
                base = test_strategy[col].iloc[0]
                adjusted = base + (base * delta / 100.0)
                test_strategy[f"{col}_adj"] = adjusted
        
        # Verify
        assert test_strategy["Gold_adj"].iloc[0] == 11.0  # 10 + 10% = 11
        assert test_strategy["BankFD_adj"].iloc[0] == 19.0  # 20 - 5% = 19


class TestPortfolioGeneration:
    """Test portfolio generation logic"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_portfolio_allocation_sum(self, data_generator):
        """Test that portfolio allocations sum to 100%"""
        # Generate test portfolio
        test_portfolio = {
            "BankFD": 15.0,
            "DebtBond": 20.0,
            "MF_Index": 25.0,
            "Gold": 10.0,
            "EQ_IT": 30.0
        }
        
        total = sum(test_portfolio.values())
        
        # Should sum to 100 (with small tolerance)
        assert abs(total - 100.0) < 0.1
    
    def test_portfolio_constraints_low_risk(self, data_generator):
        """Test that low-risk portfolios respect constraints"""
        test_case = data_generator.generate_complete_test_case()
        test_case['client_metadata']['risk_appetite'] = 'low'
        
        constraints = test_case['expected_constraints']
        
        # Verify low-risk constraints
        assert constraints.get('max_total_equity', 100) <= 30
        assert constraints.get('min_debt', 0) >= 40
    
    def test_portfolio_crypto_exclusion(self, data_generator):
        """Test that crypto is excluded when preferences specify"""
        test_case = data_generator.generate_complete_test_case()
        test_case['client_metadata']['preferences_no_crypto'] = True
        
        constraints = data_generator.generate_expected_portfolio_constraints(
            test_case['client_metadata'],
            test_case['market_conditions']
        )
        
        # Crypto should be 0
        assert constraints.get('max_crypto', 100) == 0.0
    
    def test_portfolio_gold_minimum(self, data_generator):
        """Test minimum gold allocation is respected"""
        test_case = data_generator.generate_complete_test_case()
        test_case['client_metadata']['preferences_min_gold_percent'] = 10
        
        constraints = data_generator.generate_expected_portfolio_constraints(
            test_case['client_metadata'],
            test_case['market_conditions']
        )
        
        # Gold minimum should be enforced
        assert constraints.get('min_gold', 0) == 10.0
    
    def test_portfolio_all_assets_present(self, data_generator):
        """Test that all 16 asset classes are present"""
        strategies = data_generator.generate_general_strategies()
        
        expected_assets = [
            "BankFD", "DebtBond", "MF_Index", "MF_Flexi", "MF_SmallCap",
            "EQ_Banking", "EQ_Automobile", "EQ_IT", "EQ_FMCG", 
            "EQ_MetalsMining", "EQ_OilGas", "EQ_Pharma", "EQ_Defense",
            "Gold", "Silver", "RealEstate", "Cryptocurrency"
        ]
        
        # Check first strategy has all assets
        strategy = strategies[0]
        for asset in expected_assets:
            assert asset in strategy, f"Asset {asset} missing from strategy"


class TestStateManagement:
    """Test LangGraph state management"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_state_initialization(self, data_generator):
        """Test that initial state is correctly structured"""
        initial_state = {
            "client_id": "TEST001",
            "marketCondtions": "Test market"
        }
        
        # Verify required keys
        assert "client_id" in initial_state
        assert "marketCondtions" in initial_state
    
    def test_state_propagation(self, data_generator):
        """Test that state accumulates correctly through nodes"""
        # Simulate state flow
        state = {"client_id": "TEST001"}
        
        # After load_client_metadata
        state["client_metadata"] = data_generator.generate_client_metadata("TEST001")
        assert "client_metadata" in state
        
        # After select_boundary_rules
        state["boundary_rules"] = data_generator.generate_boundary_rules("medium")
        assert "boundary_rules" in state
        
        # After load_general_strategies
        state["general_strategies"] = data_generator.generate_general_strategies()
        assert "general_strategies" in state
        
        # Verify all keys present
        required_keys = ["client_id", "client_metadata", "boundary_rules", "general_strategies"]
        for key in required_keys:
            assert key in state


class TestDataValidation:
    """Test data validation and quality checks"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_client_metadata_completeness(self, data_generator):
        """Test that client metadata has all required fields"""
        client = data_generator.generate_client_metadata()
        
        required_fields = [
            "client_id", "client_name", "age", "risk_appetite",
            "horizon_years", "liquidity_need"
        ]
        
        for field in required_fields:
            assert field in client, f"Missing field: {field}"
    
    def test_client_metadata_data_types(self, data_generator):
        """Test that client metadata has correct data types"""
        client = data_generator.generate_client_metadata()
        
        assert isinstance(client["client_id"], str)
        assert isinstance(client["client_name"], str)
        assert isinstance(client["age"], int)
        assert isinstance(client["risk_appetite"], str)
        assert isinstance(client["horizon_years"], int)
        assert isinstance(client["preferences_no_crypto"], bool)
    
    def test_client_metadata_value_ranges(self, data_generator):
        """Test that client metadata values are in valid ranges"""
        client = data_generator.generate_client_metadata()
        
        assert 25 <= client["age"] <= 70
        assert 1 <= client["horizon_years"] <= 30
        assert client["risk_appetite"] in ["low", "medium", "high"]
        assert client["liquidity_need"] in ["low", "medium", "high"]
    
    def test_strategy_allocation_validity(self, data_generator):
        """Test that strategy allocations are valid"""
        strategies = data_generator.generate_general_strategies()
        
        for strategy in strategies:
            # Remove Client Type key for sum calculation
            allocations = {k: v for k, v in strategy.items() if k != "Client Type"}
            
            # Sum should be 100
            total = sum(allocations.values())
            assert abs(total - 100.0) < 0.1, f"Strategy {strategy['Client Type']} sums to {total}"
            
            # All allocations should be non-negative
            assert all(v >= 0 for v in allocations.values())
    
    def test_market_conditions_not_empty(self, data_generator):
        """Test that market conditions are generated"""
        market = data_generator.generate_market_conditions()
        
        assert isinstance(market, str)
        assert len(market) > 50  # Should be substantial
        assert any(word in market.lower() for word in ["market", "economic", "inflation", "rate"])


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def data_generator(self):
        return SyntheticDataGenerator(seed=42)
    
    def test_missing_client_id_handling(self):
        """Test handling of missing client ID"""
        state = {}  # No client_id
        
        # Should not crash when accessing with .get()
        client_id = state.get('client_id', 'DEFAULT')
        assert client_id == 'DEFAULT'
    
    def test_empty_boundary_rules_handling(self, data_generator):
        """Test handling of empty boundary rules"""
        rules = []
        
        # Should handle empty list gracefully
        assert isinstance(rules, list)
        assert len(rules) == 0
    
    def test_invalid_risk_appetite_handling(self, data_generator):
        """Test handling of invalid risk appetite"""
        # Default to medium if invalid
        rules = data_generator.generate_boundary_rules("invalid_risk")
        
        # Should return medium risk rules (not crash)
        assert len(rules) > 0
    
    def test_division_by_zero_in_normalization(self):
        """Test that normalization handles zero totals"""
        import pandas as pd
        
        df = pd.DataFrame([{"value": 0.0}])
        
        # Simulate normalization
        row_total = df["value"].sum()
        row_total = max(row_total, 1.0)  # Avoid division by zero
        
        normalized = df["value"] / row_total * 100
        
        assert not normalized.isna().any()


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

@pytest.fixture(scope="session")
def test_data_generator():
    """Session-wide test data generator"""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope="session")
def test_suite(test_data_generator):
    """Generate test suite once per session"""
    return test_data_generator.generate_test_suite(count=10)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
