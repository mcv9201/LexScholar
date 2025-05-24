import React, { useState } from 'react';
import { Search, Moon, Sun } from 'lucide-react';

interface HeaderProps {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

const Header: React.FC<HeaderProps> = ({ isDarkMode, toggleDarkMode }) => {
  const [searchQuery, setSearchQuery] = useState('pink vigilantes india');

  return (
    <header className="sticky top-0 z-10 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <span className="text-xl font-semibold text-gray-900 dark:text-white">
              Perplexity
            </span>
          </div>

          <div className="flex-1 max-w-2xl mx-8">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-700 rounded-full bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-pink-500 sm:text-sm"
                placeholder="Ask anything..."
              />
            </div>
          </div>

          <div className="flex items-center">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-pink-500"
              aria-label="Toggle dark mode"
            >
              {isDarkMode ? (
                <Sun className="h-5 w-5" />
              ) : (
                <Moon className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;