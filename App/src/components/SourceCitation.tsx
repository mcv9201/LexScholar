import React from 'react';
import { ExternalLink, CheckCircle } from 'lucide-react';

interface SourceCitationProps {
  title: string;
  url: string;
  source: string;
  relevanceScore: number;
  contentExtracted: boolean;
}

const SourceCitation: React.FC<SourceCitationProps> = ({
  title,
  url,
  source,
  relevanceScore,
  contentExtracted,
}) => {
  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 mb-6 border border-gray-200 dark:border-gray-700">
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-gray-600 dark:text-gray-400 mb-1">
            {title}
          </h3>
          <div className="flex items-center text-xs text-gray-500 dark:text-gray-400 mb-1">
            <a 
              href={url} 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-pink-600 dark:text-pink-400 hover:underline flex items-center gap-1"
            >
              {url}
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          <div className="flex flex-wrap gap-3 mt-2 text-xs">
            <span className="text-gray-600 dark:text-gray-400">
              Source: {source}
            </span>
            <span className="text-gray-600 dark:text-gray-400">
              Relevance Score: {relevanceScore.toFixed(3)}
            </span>
            {contentExtracted && (
              <span className="flex items-center text-green-600 dark:text-green-400">
                <CheckCircle className="h-3 w-3 mr-1" />
                Content Extracted
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SourceCitation;