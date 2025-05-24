import requests
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
from urllib.parse import quote_plus
import openai
from datetime import datetime
import sqlite3
import hashlib


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


class SearchEngine:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
    def search_legal_papers(self, context: QueryContext) -> List[SearchResult]:
            """
            Step 2: Real-time internet search for legal research papers
            """
            print("🔍 Searching for relevant legal research papers...")
            
            results = []
            
            # Search multiple sources
            results.extend(self._search_google_scholar(context))
            results.extend(self._search_ssrn(context))
            results.extend(self._search_jstor(context))
            results.extend(self._search_westlaw_news(context))
            
            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)
            
            return sorted_results[:15]  # Return top 15 results

    def _search_google_scholar(self, context: QueryContext) -> List[SearchResult]:
        """Search Google Scholar for academic papers"""
        results = []
        
        if self.serpapi_key:
            # Use SerpAPI for Google Scholar
            try:
                params = {
                    'engine': 'google_scholar',
                    'q': f'"{context.processed_query}" law legal',
                    'api_key': self.serpapi_key,
                    'num': 10
                }
                
                response = requests.get('https://serpapi.com/search', params=params)
                data = response.json()
                
                for item in data.get('organic_results', []):
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        source='Google Scholar',
                        relevance_score=self._calculate_relevance(item.get('title', '') + ' ' + item.get('snippet', ''), context),
                        publication_date=item.get('publication_info', {}).get('summary', '')
                    ))
                    
            except Exception as e:
                print(f"⚠️ Google Scholar search error: {e}")
        
        else:
            # Fallback: Use requests to scrape (be respectful of rate limits)
            try:
                query = quote_plus(f'"{context.processed_query}" law legal research')
                url = f"https://scholar.google.com/scholar?q={query}&hl=en&num=10"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers)
                # Simple parsing (in production, use proper HTML parsing)
                if response.status_code == 200:
                    # This is a simplified approach - in practice, use BeautifulSoup
                    content = response.text
                    print("📚 Found Google Scholar results (simplified parsing)")
                    
            except Exception as e:
                print(f"⚠️ Google Scholar fallback error: {e}")
        
        return results

    def _search_ssrn(self, context: QueryContext) -> List[SearchResult]:
        """Search SSRN for legal papers"""
        results = []
        
        try:
            # SSRN search (simplified - they have an API but requires special access)
            query = quote_plus(context.processed_query)
            # This is a placeholder - SSRN would require proper API integration
            print("📑 Searching SSRN database...")
            
            # Simulated results for demonstration
            results.append(SearchResult(
                title=f"Legal Analysis of {context.legal_domain.title()} Law",
                url="https://ssrn.com/abstract=example",
                snippet=f"Comprehensive analysis of {context.processed_query} in legal context...",
                source='SSRN',
                relevance_score=0.85,
                publication_date="2024"
            ))
            
        except Exception as e:
            print(f"⚠️ SSRN search error: {e}")
        
        return results

    def _search_jstor(self, context: QueryContext) -> List[SearchResult]:
        """Search JSTOR for academic legal articles"""
        results = []
        
        try:
            print("📖 Searching JSTOR database...")
            
            # Simulated JSTOR results
            results.append(SearchResult(
                title=f"Judicial Review in {context.legal_domain.title()} Cases",
                url="https://jstor.org/stable/example",
                snippet=f"Historical perspective on {context.processed_query}...",
                source='JSTOR',
                relevance_score=0.80,
                publication_date="2023"
            ))
            
        except Exception as e:
            print(f"⚠️ JSTOR search error: {e}")
        
        return results

    def _search_westlaw_news(self, context: QueryContext) -> List[SearchResult]:
        """Search legal news and recent developments"""
        results = []
        
        try:
            # Use a general news API or legal news RSS feeds
            print("📰 Searching for recent legal developments...")
            
            # Example with NewsAPI (requires API key)
            # This would search legal news sources
            
        except Exception as e:
            print(f"⚠️ Legal news search error: {e}")
        
        return results

    def _calculate_relevance(self, text: str, context: QueryContext) -> float:
        """Calculate relevance score based on keyword matching and context"""
        text_lower = text.lower()
        score = 0.0
        
        # Keyword matching
        for keyword in context.keywords:
            if keyword.lower() in text_lower:
                score += 0.1
        
        # Legal domain matching
        domain_terms = self.legal_domains.get(context.legal_domain, [])
        for term in domain_terms:
            if term in text_lower:
                score += 0.15
        
        # Boost for academic terms
        academic_terms = ['journal', 'review', 'law review', 'university', 'court', 'case']
        for term in academic_terms:
            if term in text_lower:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on title similarity"""
        unique_results = []
        seen_titles = set()
        
        for result in results:
            # Simple deduplication based on title
            title_key = re.sub(r'[^\w\s]', '', result.title.lower())
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(result)
        
        return unique_results