import React from 'react';
import { ResultSection } from '../types';

interface ContentSectionProps {
  sections: ResultSection[];
}

const ContentSection: React.FC<ContentSectionProps> = ({ sections }) => {
  return (
    <div className="mb-8 animate-fadeIn">
      <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
        Key Findings
      </h2>
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
        {sections.map((section, index) => (
          <div key={index} className={index > 0 ? 'mt-6' : ''}>
            <h3 className="text-lg font-medium mb-2 text-gray-900 dark:text-white">
              {section.title}
            </h3>
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
              {section.content}
            </p>
            {index < sections.length - 1 && (
              <div className="border-b border-gray-200 dark:border-gray-700 my-4" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ContentSection;