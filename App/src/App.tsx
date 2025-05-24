import React, { useState, useEffect } from 'react';
import { Search, Settings, Mic, Volume2 } from 'lucide-react';
import { searchResults } from './data/researchData';

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="min-h-screen bg-[#1C1C1C] text-white">
      <main className="max-w-3xl mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold text-center mb-8">LexiSearch</h1>
        
        <div className="bg-[#2A2A2A] rounded-lg p-4 mb-8">
          <div className="relative mb-4">
            <input
              type="text"
              defaultValue="pink vigilantes india"
              className="w-full bg-white text-black rounded px-4 py-3 pr-12 focus:outline-none"
              placeholder="Ask anything or search for papers..."
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <button className="bg-[#2196F3] text-white px-4 py-2 rounded flex items-center gap-2">
                <Search className="w-4 h-4" />
                Search
              </button>
              <button className="bg-[#2196F3] text-white px-4 py-2 rounded flex items-center gap-2">
                Research Papers
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
        </div>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : (
          <div className="space-y-4">
            {searchResults.results.map((result, index) => (
              <div key={index} className="bg-[#2A2A2A] rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-1">{result.title}</h2>
                <p className="text-sm text-gray-400 mb-2">Source: {result.source}</p>
                <p className="text-gray-300 text-sm mb-4">{result.summary}</p>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <a 
                    href={result.url} 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:underline"
                  >
                    View Paper
                  </a>
                  <span>Relevance: {result.relevanceScore.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;