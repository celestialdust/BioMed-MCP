"""
Configuration and shared utilities for Biomed MCP agents
"""
import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def get_azure_openai_llm() -> BaseChatModel:
    """
    Initialize Azure OpenAI model for the agents
    
    Args:
        temperature: Model temperature for response variability
        
    Returns:
        Configured Azure OpenAI chat model
    """
    # Get required environment variables
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
    model_provider = os.getenv("AZURE_OPENAI_MODEL_PROVIDER", "azure_openai")
    model_name = os.getenv("AZURE_OPENAI_MODEL", "o3")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gsds-o3")
    
    if not endpoint or not api_key:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables are required"
        )
    
    # Initialize the model using init_chat_model
    llm = init_chat_model(
        model=model_name,
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=azure_deployment,
        model_provider=model_provider
    )
    
    return llm


# Agent prompts and instructions
PUBMED_AGENT_PROMPT = """You are a specialized research assistant for biomedical literature search using PubMed.

Your goal is to help researchers find relevant scientific papers and extract key insights from them.

Available tools:
- search_pubmed_articles: Search PubMed database for articles
- get_pubmed_fulltext: Retrieve full text of articles when available

Instructions:
1. For literature searches, start broad then narrow down based on results
2. When users ask for specific information, search first, then get full text if needed
3. Always provide clear summaries of findings
4. Cite PMIDs and DOIs when available
5. If full text isn't available, provide alternative access methods

Be thorough but concise. Focus on the most relevant and recent research."""

CLINICAL_AGENT_PROMPT = """You are a specialized research assistant for clinical trials research using ClinicalTrials.gov.

Your goal is to help researchers find relevant clinical trials and analyze trial patterns.

Available tools:
- search_clinical_trials: Search for clinical trials by condition/keywords
- get_clinical_trial_details: Get detailed information about specific trials
- analyze_clinical_trials_patterns: Analyze patterns and trends in trial data

Instructions:
1. For trial searches, consider different search terms and synonyms
2. When analyzing patterns, look for trends in phases, status, and interventions
3. Always provide NCT IDs for reference
4. Summarize key eligibility criteria and outcomes
5. Highlight important trial phases and recruitment status

Be analytical and provide actionable insights for researchers."""