import React from 'react';

interface ResultSummaryProps {
  summary: string;
}

const ResultSummary: React.FC<ResultSummaryProps> = ({ summary }) => {
  return (
    <div className="mb-8 animate-fadeIn">
      <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
        AI Summary
      </h2>
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
        <p className="text-gray-800 dark:text-gray-200 leading-relaxed">
          {summary}
        </p>
      </div>
    </div>
  );
};

export default ResultSummary;