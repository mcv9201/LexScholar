import React, { useState, useEffect, useRef } from 'react';
import { Search, Settings, Mic, Volume2, Upload } from 'lucide-react';
import { searchResults } from './data/researchData';

interface SearchResult {
  title: string;
  url: string;
  domain: string;
  snippet: string;
  ai_summary: string;
  relevance_score: number;
  relevance_explanation: string;
  content_extraction_success: boolean;
}

interface ApiResponse {
  success: boolean;
  message: string;
  extracted_abstract: string | null;
  total_results_found: number;
  processing_time_seconds: number;
  results: SearchResult[];
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [query, setQuery] = useState('pink vigilantes india');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [apiResults, setApiResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please select a valid PDF file');
      setSelectedFile(null);
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file first');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('research_angle', query);
    formData.append('base_paper', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/research', {
        method: 'POST',
        body: formData,
      });

      const data: ApiResponse = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Failed to perform research');
      }

      if (data.success) {
        setApiResults(data.results);
      } else {
        throw new Error(data.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setApiResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#1C1C1C] text-white">
      <main className="max-w-3xl mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold text-center mb-8">LexiSearch</h1>
        
        <div className="bg-[#2A2A2A] rounded-lg p-4 mb-8">
          <div className="relative mb-4">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full bg-white text-black rounded px-4 py-3 pr-12 focus:outline-none"
              placeholder="Ask anything or search for papers..."
            />
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex gap-2">
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="bg-[#2196F3] text-white px-4 py-2 rounded flex items-center gap-2"
              >
                <Upload className="w-4 h-4" />
                Upload PDF
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileSelect}
                className="hidden"
              />
              {selectedFile && (
                <span className="text-sm text-gray-300 py-2">
                  Selected: {selectedFile.name}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <button 
                onClick={handleSearch}
                disabled={isLoading || !selectedFile}
                className="bg-[#2196F3] text-white px-4 py-2 rounded flex items-center gap-2 disabled:opacity-50"
              >
                <Search className="w-4 h-4" />
                Search
              </button>
            </div>
            <div className="flex items-center gap-2">
              <button className="p-2 hover:bg-gray-700 rounded">
                <Settings className="w-5 h-5" />
              </button>
              <button className="p-2 hover:bg-gray-700 rounded">
                <Mic className="w-5 h-5" />
              </button>
              <button className="p-2 hover:bg-gray-700 rounded">
                <Volume2 className="w-5 h-5" />
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 text-red-400 text-sm">
              {error}
            </div>
          )}
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : (
          <div className="space-y-4">
            {apiResults.length > 0 ? (
              apiResults.map((result, index) => (
                <div key={index} className="bg-[#2A2A2A] rounded-lg p-4">
                  <h2 className="text-lg font-semibold mb-1">{result.title}</h2>
                  <p className="text-sm text-gray-400 mb-2">Source: {result.domain}</p>
                  <p className="text-gray-300 text-sm mb-4">{result.ai_summary}</p>
                  <div className="flex items-center gap-4 text-sm text-gray-400">
                    <a 
                      href={result.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:underline"
                    >
                      View Paper
                    </a>
                    <span>Relevance: {result.relevance_score.toFixed(3)}</span>
                  </div>
                  <p className="mt-4 text-sm text-gray-400">
                    {result.relevance_explanation}
                  </p>
                </div>
              ))
            ) : (
              !error && !isLoading && (
                <p className="text-center text-gray-400">
                  Upload a PDF and start your research to see results
                </p>
              )
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;