from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
import httpx
import warnings
import ssl
import urllib3
import pandas as pd
from typing import Dict, Any

models =[
"azure/genailab-maas-gpt-35-turbo",
"azure/genailab-maas-gpt-4o",
"azure/genailab-maas-gpt-4o-mini",
"azure/genailab-maas-text-embedding-3-large",
"azure/genailab-maas-whisper",
"azure_ai/genailab-maas-DeepSeek-R1",
"azure_ai/genailab-maas-DeepSeek-V3-0324",
"azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
"azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
"azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
"azure_ai/genailab-maas-Phi-3.5-vision-instruct",
"azure_ai/genailab-maas-Phi-4-reasoning"]

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
http_client = httpx.Client(verify=False)
async_http_client = httpx.AsyncClient(verify=False)

# Set your Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = "sk-N2Z4PGkQ4p8yhNfBworBww"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://genailab.tcs.in"

# Disable SSL warnings
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable SSL verification globally for all contexts
ssl._create_default_https_context = ssl._create_unverified_context

# # Monkey patch requests to disable SSL verification
import requests
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

os.environ["TIKTOKEN_CACHE_DIR"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""

# Pre-load tiktoken with SSL disabled
try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    import functools
    
    # Patch tiktoken's load function to use unverified context
    original_load = load_tiktoken_bpe
    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        import urllib.request
        old_opener = urllib.request.urlopen
        def new_opener(url, *args, **kwargs):
            context = ssl._create_unverified_context()
            return old_opener(url, *args, context=context, **kwargs)
        urllib.request.urlopen = new_opener
        result = original_load(*args, **kwargs)
        urllib.request.urlopen = old_opener
        return result
    tiktoken.load.load_tiktoken_bpe = patched_load
except Exception as e:
    print(f"Tiktoken patch warning: {e}")

# Create custom HTTP client with SSL verification disabled
http_client = httpx.Client(verify=False, timeout=60.0)
async_http_client = httpx.AsyncClient(verify=False, timeout=60.0)

# Initialize Azure OpenAI LLM with custom HTTP client
# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment="azure/genailab-maas-gpt-4o",  # e.g., "gpt-4" or "gpt-35-turbo"
    api_version="2024-02-15-preview",
    temperature=0.7,
    max_tokens=4096,
    http_client=http_client,
    http_async_client=async_http_client
)

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="azure/genailab-maas-text-embedding-3-large",  # e.g., "text-embedding-ada-002"
    api_version="2024-02-15-preview",
    http_client=http_client,
    http_async_client=async_http_client
)


os.environ["TAVILY_API_KEY"] = "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"

from langchain_community.retrievers import TavilySearchAPIRetriever

serachRe = TavilySearchAPIRetriever(k=2,    
    search_depth="advanced",
    # include_images=True,
    news=True,
    include_domains=[
        "economictimes.indiatimes.com",
        "moneycontrol.com",
        "business-standard.com",
        "bloomberg.com",
        "livemint.com",
        "finance.yahoo.com",
        "cnbc.com",
        "nse.com",
        "bse.com",
    ])

# Test the LLM
print("Testing LLM...")
try:
    response = llm.invoke("Hello, how are you?")
    print(f"LLM Response: {response.content}")
    print("✓ LLM working successfully!")
except Exception as e:
    print(f"✗ LLM Error: {e}")

# Test the embeddings
print("\nTesting Embeddings...")
try:
    text = "This is a test sentence."
    embedding_vector = embeddings.embed_query(text)
    print(f"Embedding dimension: {len(embedding_vector)}")
    print(f"First 5 values: {embedding_vector[:5]}")
    print("✓ Embeddings working successfully!")
except Exception as e:
    print(f"✗ Embedding Error: {e}")

class DataQualityFramework:
    """
    Simple Data Quality Framework for validating pandas DataFrames.
    Extend this class for more advanced checks or integrate with libraries like Great Expectations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def check_nulls(self) -> Dict[str, int]:
        nulls = self.df.isnull().sum().to_dict()
        self.results['nulls'] = nulls
        return nulls

    def check_duplicates(self) -> int:
        dups = self.df.duplicated().sum()
        self.results['duplicates'] = dups
        return dups

    def check_schema(self, expected_columns: list) -> bool:
        actual = set(self.df.columns)
        expected = set(expected_columns)
        schema_ok = actual == expected
        self.results['schema'] = schema_ok
        return schema_ok

    def run_all_checks(self, expected_columns: list = None) -> Dict[str, Any]:
        self.check_nulls()
        self.check_duplicates()
        if expected_columns:
            self.check_schema(expected_columns)
        return self.results

# Example usage (uncomment and adapt as needed):
df = pd.read_csv('client.csv')
dq = DataQualityFramework(df)
dq_results = dq.run_all_checks(expected_columns=["client_id", "name", ...])
print("Data Quality Results:", dq_results)

