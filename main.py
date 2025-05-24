#!/usr/bin/env python3
"""
FastAPI Legal Research Service

API for AI-powered legal research with PDF upload support.
Extracts text from PDF, identifies abstract, and performs research.

Required packages: 
fastapi uvicorn python-multipart PyPDF2 pdfplumber openai requests scikit-learn numpy

Run with: uvicorn main:app --reload
"""

import os
import io
import re
import json
import requests
import numpy as np
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF processing imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available. Install with: pip install pdfplumber")

# Legal research imports
from sklearn.metrics.pairwise import cosine_similarity
import openai
import time
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional web scraping imports
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Legal Research API",
    description="AI-powered legal research with PDF upload support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ResearchRequest(BaseModel):
    research_angle: str
    base_paper_text: Optional[str] = None

class SearchResultResponse(BaseModel):
    title: str
    url: str
    domain: str
    snippet: str
    ai_summary: str
    relevance_score: float
    relevance_explanation: str
    content_extraction_success: bool

class ResearchResponse(BaseModel):
    success: bool
    message: str
    extracted_abstract: Optional[str] = None
    total_results_found: int
    processing_time_seconds: float
    results: List[SearchResultResponse]

# Configuration
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.pdf'}

config = Config()

@dataclass
class SearchResult:
    """Data class to store search result metadata"""
    title: str
    url: str
    snippet: str
    domain: str
    full_content: str = ""
    ai_summary: str = ""
    relevance_score: float = 0.0
    relevance_explanation: str = ""
    content_extraction_success: bool = False

class PDFProcessor:
    """Handle PDF text extraction and abstract identification"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file: bytes) -> str:
        """
        Extract text from PDF using multiple methods
        
        Args:
            pdf_file: PDF file as bytes
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Method 1: Try pdfplumber first (better for complex layouts)
        if PDFPLUMBER_AVAILABLE:
            try:
                with io.BytesIO(pdf_file) as pdf_stream:
                    with pdfplumber.open(pdf_stream) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                
                if len(text.strip()) > 100:
                    return text.strip()
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
        
        # Method 2: Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                with io.BytesIO(pdf_file) as pdf_stream:
                    pdf_reader = PyPDF2.PdfReader(pdf_stream)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if len(text.strip()) > 100:
                    return text.strip()
            except Exception as e:
                print(f"PyPDF2 extraction failed: {e}")
        
        if not text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF. Please ensure the PDF contains readable text."
            )
        
        return text.strip()
    
    @staticmethod
    def extract_abstract(full_text: str, openai_client) -> str:
        """
        Extract abstract section from full paper text using AI
        
        Args:
            full_text: Full paper text
            openai_client: OpenAI client instance
            
        Returns:
            Extracted abstract text
        """
        # First try rule-based extraction
        abstract = PDFProcessor._rule_based_abstract_extraction(full_text)
        if abstract:
            return abstract
        
        # Fallback to AI-based extraction
        return PDFProcessor._ai_based_abstract_extraction(full_text, openai_client)
    
    @staticmethod
    def _rule_based_abstract_extraction(text: str) -> Optional[str]:
        """
        Extract abstract using common patterns
        
        Args:
            text: Full paper text
            
        Returns:
            Abstract text if found, None otherwise
        """
        # Common abstract patterns
        patterns = [
            r'(?i)abstract\s*[:.]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.|I\.|background))',
            r'(?i)abstract\s*\n\s*(.*?)(?=\n\s*\n|\n\s*[A-Z][A-Za-z\s]*:)',
            r'(?i)summary\s*[:.]?\s*(.*?)(?=\n\s*(?:keywords?|introduction|1\.))'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                abstract = matches.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)
                if 50 <= len(abstract) <= 2000:  # Reasonable abstract length
                    return abstract
        
        return None
    
    @staticmethod
    def _ai_based_abstract_extraction(text: str, openai_client) -> str:
        """
        Extract abstract using AI when rule-based fails
        
        Args:
            text: Full paper text
            openai_client: OpenAI client
            
        Returns:
            Abstract text
        """
        # Truncate text for API limits
        truncated_text = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""
        Extract the abstract or summary from this legal research paper. If there's no explicit abstract section, 
        create a concise summary (150-300 words) of the main arguments, findings, and conclusions.
        
        Paper text:
        {truncated_text}
        
        Return only the abstract/summary text without any labels or formatting.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            abstract = response.choices[0].message.content.strip()
            return abstract
            
        except Exception as e:
            print(f"AI abstract extraction failed: {e}")
            # Return first 500 words as fallback
            words = text.split()[:500]
            return " ".join(words) + "..."

class LegalResearchPipeline:
    """Legal research pipeline adapted for FastAPI"""
    
    def __init__(self, openai_api_key: str, serper_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.serper_api_key = serper_api_key
        
        self.trusted_domains = {
            'livelaw.in', 'indiankanoon.org', 'scobserver.in', 'nlsir.in',
            'barandbench.com', 'manupatra.com', 'lexforti.com', 'lawctopus.com',
            'latestlaws.com', 'advocatekhoj.com'
        }
    
    def _build_site_constraint(self) -> str:
        sites = " OR ".join([f"site:{domain}" for domain in self.trusted_domains])
        return f"({sites})"
    
    async def generate_search_queries(self, base_paper: str, research_angle: str, num_queries: int = 5) -> List[str]:
        """Generate search queries asynchronously"""
        site_constraint = self._build_site_constraint()
        
        prompt = f"""
        Based on this legal research context, generate {num_queries} diverse Google search queries using operators (AND, OR, quotes).
        
        Base Paper Abstract: {base_paper[:1000]}...
        Research Angle: {research_angle}
        
        Generate queries that:
        1. Use Google search operators effectively
        2. Include relevant legal terminology
        3. Cover different aspects of the research angle
        4. Balance specificity with comprehensiveness
        
        Do NOT include site: operators - they will be added automatically.
        Return only the search queries, one per line.
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
            )
            
            base_queries = response.choices[0].message.content.strip().split('\n')
            base_queries = [q.strip().strip('"').strip("'") for q in base_queries if q.strip()]
            
            # Add site constraints
            constrained_queries = [f"({query}) AND {site_constraint}" for query in base_queries[:num_queries]]
            return constrained_queries
            
        except Exception as e:
            print(f"Error generating search queries: {e}")
            return [f'"{research_angle}" AND {site_constraint}']
    
    async def search_web(self, query: str, num_results: int = 8) -> List[Dict]:
        """Search web asynchronously"""
        url = "https://google.serper.dev/search"
        
        payload = {
            "q": query,
            "num": num_results,
            "gl": "in",
            "hl": "en"
        }
        
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(url, json=payload, headers=headers, timeout=10)
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get('organic', [])
            
            # Filter by trusted domains
            verified_results = []
            for result in results:
                domain = self._extract_domain(result.get('link', ''))
                if domain in self.trusted_domains or not domain:
                    verified_results.append(result)
            
            return verified_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower().replace('www.', '')
        except:
            return ""
    
    async def generate_ai_summary(self, title: str, content: str, research_context: str) -> str:
        """Generate AI summary asynchronously"""
        max_content_length = 1500
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""
        Summarize this legal article focusing on aspects relevant to the research context.
        
        Research Context: {research_context}
        Article Title: {title}
        Content: {content}
        
        Create a 100-200 word summary covering:
        1. Main legal arguments and conclusions
        2. Relevant legal principles, cases, or precedents
        3. Practical implications
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"AI summary error: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def generate_relevance_explanation(self, result: SearchResult, base_paper: str, research_angle: str) -> str:
        """Generate relevance explanation asynchronously"""
        content_for_analysis = result.ai_summary if result.ai_summary else result.snippet
        
        prompt = f"""
        Explain why this search result is relevant to the user's research (3-4 sentences).
        
        User's Research: {research_angle}
        Base Paper: {base_paper[:500]}...
        
        Result: {result.title}
        Content: {content_for_analysis}
        
        Focus on specific legal connections and substantive relevance.
        Start with "This result is relevant because..."
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Relevance explanation error: {e}")
            return f"This result is relevant (score: {result.relevance_score:.3f}) to your research on {research_angle.split(',')[0]}."
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings synchronously"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return np.array([item.embedding for item in response.data])
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.zeros((len(texts), 1536))
    
    async def run_research(self, base_paper: str, research_angle: str, top_k: int = 6) -> List[SearchResult]:
        """Run the complete research pipeline asynchronously"""
        # Generate search queries
        queries = await self.generate_search_queries(base_paper, research_angle)
        
        # Search web for all queries
        all_search_results = []
        search_tasks = [self.search_web(query) for query in queries]
        search_results_list = await asyncio.gather(*search_tasks)
        
        for results in search_results_list:
            all_search_results.extend(results)
        
        # Process results
        processed_results = []
        seen_urls = set()
        
        for result in all_search_results:
            url = result.get('link', '')
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            domain = self._extract_domain(url)
            
            if title and snippet and domain:
                search_result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    domain=domain
                )
                processed_results.append(search_result)
        
        # Generate AI summaries
        summary_tasks = [
            self.generate_ai_summary(result.title, result.snippet, f"{base_paper[:500]}... Research: {research_angle}")
            for result in processed_results
        ]
        summaries = await asyncio.gather(*summary_tasks)
        
        for result, summary in zip(processed_results, summaries):
            result.ai_summary = summary
            result.content_extraction_success = True  # Simplified for API
        
        # Compute relevance scores
        if processed_results:
            user_context = f"Base Paper: {base_paper}\nResearch Angle: {research_angle}"
            result_texts = [f"{r.title} {r.ai_summary}" for r in processed_results]
            all_texts = [user_context] + result_texts
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.compute_embeddings, 
                all_texts
            )
            
            if embeddings.shape[0] > 0:
                context_embedding = embeddings[0:1]
                result_embeddings = embeddings[1:]
                similarities = cosine_similarity(context_embedding, result_embeddings)[0]
                
                for i, result in enumerate(processed_results):
                    result.relevance_score = float(similarities[i])
        
        # Sort by relevance
        processed_results.sort(key=lambda x: x.relevance_score, reverse=True)
        top_results = processed_results[:top_k]
        
        # Generate relevance explanations for top results
        explanation_tasks = [
            self.generate_relevance_explanation(result, base_paper, research_angle)
            for result in top_results
        ]
        explanations = await asyncio.gather(*explanation_tasks)
        
        for result, explanation in zip(top_results, explanations):
            result.relevance_explanation = explanation
        
        return top_results

# Initialize global pipeline instance
pipeline = None
def get_pipeline():
    global pipeline
    if pipeline is None:
        if not config.OPENAI_API_KEY or not config.SERPER_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="API keys not configured. Set OPENAI_API_KEY and SERPER_API_KEY environment variables."
            )
        pipeline = LegalResearchPipeline(config.OPENAI_API_KEY, config.SERPER_API_KEY)
    return pipeline

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Legal Research API",
        "version": "1.0.0",
        "endpoints": {
            "research": "/research",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "openai": bool(config.OPENAI_API_KEY),
            "serper": bool(config.SERPER_API_KEY),
            "pdf_processing": PDFPLUMBER_AVAILABLE or PYPDF2_AVAILABLE
        }
    }

@app.post("/research", response_model=ResearchResponse)
async def research_legal_documents(
    research_angle: str = Form(..., description="The research angle or question you want to explore"),
    base_paper: UploadFile = File(..., description="PDF file of the base research paper")
):
    """
    Main endpoint for legal research
    
    Upload a PDF paper and specify research angle to get relevant legal documents
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not base_paper.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read PDF content
        pdf_content = await base_paper.read()
        if len(pdf_content) > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE/1024/1024}MB")
        
        # Extract text from PDF
        try:
            full_text = PDFProcessor.extract_text_from_pdf(pdf_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
        
        # Extract abstract
        research_pipeline = get_pipeline()
        abstract = PDFProcessor.extract_abstract(full_text, research_pipeline.openai_client)
        
        # Run research pipeline
        results = await research_pipeline.run_research(abstract, research_angle, top_k=8)
        
        # Convert results to response format
        result_responses = []
        for result in results:
            result_responses.append(SearchResultResponse(
                title=result.title,
                url=result.url,
                domain=result.domain,
                snippet=result.snippet,
                ai_summary=result.ai_summary,
                relevance_score=result.relevance_score,
                relevance_explanation=result.relevance_explanation,
                content_extraction_success=result.content_extraction_success
            ))
        
        processing_time = time.time() - start_time
        
        return ResearchResponse(
            success=True,
            message=f"Successfully processed research request. Found {len(results)} relevant results.",
            extracted_abstract=abstract[:500] + "..." if len(abstract) > 500 else abstract,
            total_results_found=len(results),
            processing_time_seconds=round(processing_time, 2),
            results=result_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        return ResearchResponse(
            success=False,
            message=f"Research pipeline failed: {str(e)}",
            extracted_abstract=None,
            total_results_found=0,
            processing_time_seconds=round(processing_time, 2),
            results=[]
        )

@app.post("/research-text", response_model=ResearchResponse)
async def research_with_text(request: ResearchRequest):
    """
    Alternative endpoint for research with text input instead of PDF upload
    """
    start_time = time.time()
    
    try:
        if not request.base_paper_text:
            raise HTTPException(status_code=400, detail="base_paper_text is required")
        
        # Use provided text directly as abstract
        abstract = request.base_paper_text
        
        # Run research pipeline
        research_pipeline = get_pipeline()
        results = await research_pipeline.run_research(abstract, request.research_angle, top_k=8)
        
        # Convert results to response format
        result_responses = []
        for result in results:
            result_responses.append(SearchResultResponse(
                title=result.title,
                url=result.url,
                domain=result.domain,
                snippet=result.snippet,
                ai_summary=result.ai_summary,
                relevance_score=result.relevance_score,
                relevance_explanation=result.relevance_explanation,
                content_extraction_success=result.content_extraction_success
            ))
        
        processing_time = time.time() - start_time
        
        return ResearchResponse(
            success=True,
            message=f"Successfully processed research request. Found {len(results)} relevant results.",
            extracted_abstract=abstract[:500] + "..." if len(abstract) > 500 else abstract,
            total_results_found=len(results),
            processing_time_seconds=round(processing_time, 2),
            results=result_responses
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        return ResearchResponse(
            success=False,
            message=f"Research pipeline failed: {str(e)}",
            extracted_abstract=None,
            total_results_found=0,
            processing_time_seconds=round(processing_time, 2),
            results=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
