import requests
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
from urllib.parse import quote_plus
from groq import Groq
from datetime import datetime
from search_engine import SearchEngine

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    publication_date: Optional[str] = None

@dataclass
class QueryContext:
    original_query: str
    processed_query: str
    intent: str
    legal_domain: str
    keywords: List[str]
    timestamp: datetime

class LegalAIAgent:
    def __init__(self, groq_api_key: str, serpapi_key: str):
        
        self.groq_client = Groq(api_key=groq_api_key)
        # self.serpapi_key = serpapi_key
        self.search_engine = SearchEngine(serpapi_key)
        
        # Legal domains for context understanding
        self.legal_domains = {
            'constitutional': ['constitution', 'constitutional law', 'civil rights', 'bill of rights'],
            'criminal': ['criminal law', 'criminal procedure', 'evidence', 'sentencing'],
            'civil': ['tort', 'contract', 'property', 'civil procedure'],
            'corporate': ['corporate law', 'securities', 'mergers', 'business law'],
            'intellectual_property': ['patent', 'trademark', 'copyright', 'ip law'],
            'family': ['family law', 'divorce', 'custody', 'adoption'],
            'environmental': ['environmental law', 'climate', 'pollution', 'sustainability'],
            'tax': ['tax law', 'taxation', 'irs', 'tax code'],
            'labor': ['employment law', 'labor relations', 'workplace', 'discrimination'],
            'international': ['international law', 'treaty', 'human rights', 'diplomatic']
        }

    def understand_query(self, query: str) -> QueryContext:
        """
        Step 1: Advanced query understanding using LLM
        """
        print("🧠 Understanding your legal query...")
        
        # Use GPT for query understanding
        system_prompt = """
        You are a legal research AI assistant. Analyze the user's query and provide:
        1. Intent: What type of legal information they're seeking
        2. Legal Domain: Which area of law this relates to
        3. Keywords: Key terms for searching legal databases
        4. Processed Query: An optimized search query for legal research

        Respond in JSON format:
        {
            "intent": "brief description of what user wants",
            "legal_domain": "primary area of law",
            "keywords": ["key", "terms", "for", "search"],
            "processed_query": "optimized search query",
            "search_suggestions": ["alternative", "search", "terms"]
        }
        """
        
        # try:
        response = self.groq_client.chat.completions.create(
            model="qwen-qwq-32b",  # Groq's fast model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this legal query: {query}"}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        
        analysis = json.loads(response.choices[0].message.content)
        
        context = QueryContext(
            original_query=query,
            processed_query=analysis['processed_query'],
            intent=analysis['intent'],
            legal_domain=analysis['legal_domain'],
            keywords=analysis['keywords'],
            timestamp=datetime.now()
        )
            
            
        return context
            
        # except Exception as e:
        #     print(f"⚠️ Error in query understanding: {e}")
        #     # Fallback to basic processing
        #     return self._basic_query_processing(query)

    def _basic_query_processing(self, query: str) -> QueryContext:
        """Fallback query processing without LLM"""
        # Simple keyword extraction
        keywords = re.findall(r'\b\w+\b', query.lower())
        keywords = [k for k in keywords if len(k) > 3 and k not in ['what', 'how', 'when', 'where', 'why']]
        
        # Determine legal domain
        legal_domain = 'general'
        for domain, terms in self.legal_domains.items():
            if any(term in query.lower() for term in terms):
                legal_domain = domain
                break
        
        return QueryContext(
            original_query=query,
            processed_query=query,
            intent="Find legal research papers",
            legal_domain=legal_domain,
            keywords=keywords[:10],  # Limit to 10 keywords
            timestamp=datetime.now()
        )

    #search_legal_papers code here

    def search(self, query: str) -> Dict:
        """
        Main search function that combines query understanding and search
        """
        print(f"🔍 Processing query: '{query}'")
        print("-" * 50)
        
        # Step 1: Understand the query
        context = self.understand_query(query)
        print(f"📋 Intent: {context.intent}")
        print(f"🏛️ Legal Domain: {context.legal_domain}")
        print(f"🔑 Keywords: {', '.join(context.keywords)}")
        print("-" * 50)
        
        # Step 2: Search for papers
        results = self.search_engine.search_legal_papers(context)
        
        # Store results in conversation history
        # self.conversation_history.append({
        #     'query': query,
        #     'context': context,
        #     'results': results,
        #     'timestamp': datetime.now()
        # })
        
        return {
            'query_context': context,
            'results': results,
            'total_found': len(results)
        }

    def format_results(self, search_response: Dict) -> str:
        """Format search results for display"""
        context = search_response['query_context']
        results = search_response['results']
        
        output = f"\n📊 LEGAL RESEARCH RESULTS\n"
        output += f"Query: {context.original_query}\n"
        output += f"Legal Domain: {context.legal_domain.title()}\n"
        output += f"Found: {len(results)} relevant papers\n"
        output += "=" * 60 + "\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result.title}\n"
            output += f"   Source: {result.source}\n"
            output += f"   URL: {result.url}\n"
            output += f"   Relevance: {result.relevance_score:.2f}\n"
            if result.publication_date:
                output += f"   Published: {result.publication_date}\n"
            output += f"   Summary: {result.snippet}\n"
            output += "-" * 40 + "\n"
        
        return output

# Example usage and demo
def main():
    """
    Demo function showing how to use the Legal AI Agent
    """
    # Initialize the agent (you'll need to provide API keys)
    agent = LegalAIAgent(
        groq_api_key="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",
        serpapi_key="3611eaea5638a59ec95b6329077ddd9c8a71ece3"  # Optional
    )
    
    # Example queries
    test_queries = [
        "What are the latest developments in AI liability law?",
        "Find research papers on Fourth Amendment and digital privacy",
        "Contract law cases involving force majeure during COVID-19",
        "Environmental law and carbon credit regulations",
        "Criminal procedure and Miranda rights exceptions"
    ]
    
    print("🤖 Legal AI Research Agent Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Testing Query: {query}")
        results = agent.search(query)
        formatted = agent.format_results(results)
        print(formatted)
        time.sleep(2)  # Be respectful with API calls

if __name__ == "__main__":
    # For interactive use
    print("🤖 Legal AI Research Agent")
    print("Enter your legal research queries (type 'quit' to exit)")
    
    # You would initialize with your actual API keys
    agent = LegalAIAgent(
        groq_api_key="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",
        serpapi_key="3611eaea5638a59ec95b6329077ddd9c8a71ece3" 
    )
    
    while True:
        query = input("\n💼 Your legal research query: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            results = agent.search(query)
            print(agent.format_results(results))
        except Exception as e:
            print(f"⚠️ Error: {e}")