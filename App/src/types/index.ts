export interface ResearchResult {
  title: string;
  url: string;
  source: string;
  relevanceScore: number;
  contentExtracted: boolean;
  summary: string;
}

export interface SearchResult {
  results: ResearchResult[];
}