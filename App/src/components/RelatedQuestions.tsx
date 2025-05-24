import React from 'react';
import { ArrowRight } from 'lucide-react';
import { RelatedQuestion } from '../types';

interface RelatedQuestionsProps {
  questions: RelatedQuestion[];
}

const RelatedQuestions: React.FC<RelatedQuestionsProps> = ({ questions }) => {
  return (
    <div className="mb-8 animate-fadeIn">
      <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
        Related Questions
      </h2>
      <div className="grid gap-3">
        {questions.map((question) => (
          <button
            key={question.id}
            className="flex items-center justify-between p-4 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors"
          >
            <span className="text-gray-800 dark:text-gray-200 text-left">
              {question.text}
            </span>
            <ArrowRight className="h-4 w-4 text-gray-400" />
          </button>
        ))}
      </div>
    </div>
  );
};

export default RelatedQuestions;