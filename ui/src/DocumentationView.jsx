import React, { useState, useMemo, useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import {
  Search,
  ShieldCheck,
  Cpu,
  Check,
  Copy,
  ChevronRight,
  ChevronDown,
  AlertTriangle,
  FileCode,
  Layers,
  Terminal,
  Hash,
  ArrowLeft,
  ArrowRight,
  Activity,
  Eye,
  Scan,
  Database,
  FileText,
  Sparkles,
  HelpCircle,
  X,
  BookOpen
} from 'lucide-react';
import { DOC_SECTIONS } from './docsData';



// Component for rendering publication-grade mathematical formulas with KaTeX
function MathFormulaViewer({ formula }) {
  const [copied, setCopied] = useState(false);

  const renderedHtml = useMemo(() => {
    if (!formula) return null;
    try {
      return katex.renderToString(formula, {
        displayMode: true,
        throwOnError: false,
        output: 'htmlAndMathml'
      });
    } catch (e) {
      return null;
    }
  }, [formula]);

  const handleCopy = () => {
    if (!formula) return;
    navigator.clipboard.writeText(formula);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="docs-math-card">
      <div className="docs-math-header">
        <div className="math-header-title">
          <Terminal size={12} className="text-accent" />
          <span>FORMAL MATHEMATICAL SPECIFICATION</span>
        </div>
        <button
          className="docs-formula-copy-btn"
          onClick={handleCopy}
          title="Copy Formula Notation"
        >
          {copied ? (
            <>
              <Check size={12} className="text-success" />
              <span>Copied Formula</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy Formula</span>
            </>
          )}
        </button>
      </div>

      <div className="docs-math-body">
        {renderedHtml ? (
          <div
            className="katex-display-container"
            dangerouslySetInnerHTML={{ __html: renderedHtml }}
          />
        ) : (
          <code className="fallback-math-code">{formula}</code>
        )}
      </div>
    </div>
  );
}

export default function DocumentationView() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategoryFilter, setActiveCategoryFilter] = useState('all');
  const [selectedTopicId, setSelectedTopicId] = useState('lex-ece');
  const [copiedCode, setCopiedCode] = useState(false);

  // ALL COLLAPSED BY DEFAULT as requested
  const [expandedSections, setExpandedSections] = useState({});

  const contentPaneRef = useRef(null);

  // Flatten all topics for easy index lookup (prev/next navigation)
  const allTopics = useMemo(() => {
    const list = [];
    DOC_SECTIONS.forEach((section) => {
      section.items.forEach((item) => {
        list.push({
          ...item,
          sectionId: section.id,
          sectionTitle: section.title
        });
      });
    });
    return list;
  }, []);

  // Filter sections and items based on search and category filter
  const filteredSections = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return DOC_SECTIONS.map((section) => {
      // Filter by category pill if active
      if (activeCategoryFilter !== 'all' && section.id !== activeCategoryFilter) {
        return null;
      }

      const matchingItems = section.items.filter((item) => {
        if (!query) return true;
        return (
          item.title.toLowerCase().includes(query) ||
          (item.overview && item.overview.toLowerCase().includes(query)) ||
          (item.type && item.type.toLowerCase().includes(query)) ||
          (item.tag && item.tag.toLowerCase().includes(query)) ||
          (item.formula && item.formula.toLowerCase().includes(query)) ||
          (item.failureMode && item.failureMode.toLowerCase().includes(query))
        );
      });

      if (matchingItems.length === 0) return null;

      return {
        ...section,
        items: matchingItems
      };
    }).filter(Boolean);
  }, [searchQuery, activeCategoryFilter]);

  // Current selected item
  const currentTopic = useMemo(() => {
    return allTopics.find((t) => t.id === selectedTopicId) || allTopics[0];
  }, [allTopics, selectedTopicId]);

  // Current index for prev / next
  const currentIndex = useMemo(() => {
    return allTopics.findIndex((t) => t.id === currentTopic.id);
  }, [allTopics, currentTopic]);

  const prevTopic = currentIndex > 0 ? allTopics[currentIndex - 1] : null;
  const nextTopic = currentIndex < allTopics.length - 1 ? allTopics[currentIndex + 1] : null;

  // Automatically expand a section when searching so results are immediately visible
  useEffect(() => {
    if (searchQuery.trim()) {
      const autoExpanded = {};
      filteredSections.forEach((s) => {
        autoExpanded[s.id] = true;
      });
      setExpandedSections(autoExpanded);
    }
  }, [searchQuery, filteredSections]);

  // Scroll to top of content pane whenever selected topic changes
  useEffect(() => {
    if (contentPaneRef.current) {
      contentPaneRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [selectedTopicId]);

  const handleSelectTopic = (topicId, sectionId) => {
    setSelectedTopicId(topicId);
    if (sectionId) {
      setExpandedSections((prev) => ({
        ...prev,
        [sectionId]: true
      }));
    }
  };

  const handleCopyCode = (code) => {
    if (!code) return;
    navigator.clipboard.writeText(code);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const toggleSection = (sectionId) => {
    setExpandedSections((prev) => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const expandAll = () => {
    const updated = {};
    DOC_SECTIONS.forEach((s) => (updated[s.id] = true));
    setExpandedSections(updated);
  };

  const collapseAll = () => {
    setExpandedSections({});
  };

  return (
    <div className="language-docs-container">
      {/* Top Header Bar with TrustCheck Logo */}
      <header className="language-docs-header">
        <div className="docs-header-left">
          <div className="docs-header-brand">
            <img src="/logo.png" alt="TrustCheck" className="docs-header-brand-logo" />
            <span className="docs-header-title">TrustCheck Reference Documentation</span>
          </div>
          <span className="docs-header-stat">{allTopics.length} Methods &amp; Metrics</span>
        </div>

        <div className="docs-header-actions">
          <button className="docs-action-btn" onClick={expandAll} title="Expand All Sections">
            Expand All
          </button>
          <button className="docs-action-btn" onClick={collapseAll} title="Collapse All Sections">
            Collapse All
          </button>
        </div>
      </header>

      <div className="language-docs-body">
        {/* LEFT COLUMN: Sidebar Topics Selector */}
        <aside className="docs-sidebar-column">
          {/* Search Box */}
          <div className="docs-sidebar-search">
            <Search size={15} className="search-icon" />
            <input
              type="text"
              placeholder="Search 73 topics, formulas, metrics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="docs-search-input"
            />
            {searchQuery && (
              <button
                className="docs-clear-btn"
                onClick={() => setSearchQuery('')}
                title="Clear search"
              >
                <X size={13} />
              </button>
            )}
          </div>

          {/* Quick Filter Bar */}
          <div className="docs-filter-tabs">
            <button
              className={`docs-filter-tab ${activeCategoryFilter === 'all' ? 'active' : ''}`}
              onClick={() => setActiveCategoryFilter('all')}
            >
              All ({allTopics.length})
            </button>
            <button
              className={`docs-filter-tab ${activeCategoryFilter === 'lexicon' ? 'active' : ''}`}
              onClick={() => setActiveCategoryFilter('lexicon')}
            >
              Lexicon
            </button>
            <button
              className={`docs-filter-tab ${activeCategoryFilter === 'universal' ? 'active' : ''}`}
              onClick={() => setActiveCategoryFilter('universal')}
            >
              Governance
            </button>
          </div>

          {/* Topics Accordion List */}
          <nav className="docs-topics-nav">
            {filteredSections.length === 0 ? (
              <div className="docs-no-results">
                <AlertTriangle size={18} />
                <span>No topics match "{searchQuery}"</span>
              </div>
            ) : (
              filteredSections.map((section) => {
                // Collapsed by default unless toggled in state or searching
                const isExpanded = !!expandedSections[section.id];

                return (
                  <div key={section.id} className="docs-section-group">
                    <button
                      className={`docs-section-header-btn ${isExpanded ? 'expanded' : ''}`}
                      onClick={() => toggleSection(section.id)}
                    >
                      <div className="section-title-wrap">
                        <ChevronRight
                          size={12}
                          className={`section-chevron ${isExpanded ? 'rotated' : ''}`}
                        />
                        <span className="section-title">{section.title}</span>
                      </div>
                      <div className="section-meta-wrap">
                        <span className="section-count">{section.items.length}</span>
                      </div>
                    </button>

                    {isExpanded && (
                      <ul className="docs-section-items-list">
                        {section.items.map((item) => {
                          const isSelected = item.id === currentTopic.id;
                          return (
                            <li key={item.id}>
                              <button
                                className={`docs-topic-link ${isSelected ? 'active' : ''}`}
                                onClick={() => handleSelectTopic(item.id, section.id)}
                              >
                                <span className="topic-dot" />
                                <span className="topic-link-text">{item.title}</span>
                                {item.tag && (
                                  <span className="topic-mini-tag">
                                    {item.tag.replace('METHOD ', 'M')}
                                  </span>
                                )}
                              </button>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                );
              })
            )}
          </nav>
        </aside>

        {/* RIGHT COLUMN: Selected Topic Explanation Pane */}
        <main className="docs-content-column" ref={contentPaneRef}>
          {currentTopic && (
            <article className="docs-article">
              {/* Breadcrumb Path */}
              <nav className="docs-breadcrumb">
                <span className="breadcrumb-root">Documentation</span>
                <ChevronRight size={12} className="breadcrumb-separator" />
                <span className="breadcrumb-section">{currentTopic.sectionTitle}</span>
                <ChevronRight size={12} className="breadcrumb-separator" />
                <span className="breadcrumb-current">{currentTopic.title}</span>
              </nav>

              {/* Title & Meta Header */}
              <header className="docs-article-header">
                <div className="docs-title-row">
                  <h1 className="docs-topic-heading">{currentTopic.title}</h1>
                  <div className="docs-badges-row">
                    {currentTopic.tag && (
                      <span className="docs-badge tag-badge">{currentTopic.tag}</span>
                    )}
                    {currentTopic.type && (
                      <span className="docs-badge type-badge">{currentTopic.type}</span>
                    )}
                    <span className="docs-badge section-badge">{currentTopic.sectionTitle}</span>
                  </div>
                </div>
              </header>

              {/* Topic Content Sections */}
              <div className="docs-article-content">
                {/* 1. Overview & Specification */}
                <section className="docs-section-block">
                  <h2 className="docs-section-heading">
                    <BookOpen size={16} className="heading-icon text-primary" />
                    Overview &amp; Technical Specification
                  </h2>
                  <div className="docs-prose">
                    <p>{currentTopic.overview}</p>
                  </div>
                </section>

                {/* 2. Mathematical Formulation / Formal Notation (Rendered with KaTeX) */}
                {currentTopic.formula && (
                  <section className="docs-section-block">
                    <h2 className="docs-section-heading">
                      <Terminal size={16} className="heading-icon text-accent" />
                      Mathematical Formulation &amp; Bound Specification
                    </h2>
                    <MathFormulaViewer formula={currentTopic.formula} />
                  </section>
                )}

                {/* 3. Clinical & Operational Rationale */}
                {currentTopic.clinicalRationale && (
                  <section className="docs-section-block">
                    <h2 className="docs-section-heading">
                      <Activity size={16} className="heading-icon text-info" />
                      Clinical &amp; Operational Context
                    </h2>
                    <div className="docs-prose docs-callout-info">
                      <p>{currentTopic.clinicalRationale}</p>
                    </div>
                  </section>
                )}

                {/* 4. Failure Mode Prevented / Safety Alert */}
                {currentTopic.failureMode && (
                  <section className="docs-section-block">
                    <h2 className="docs-section-heading">
                      <ShieldCheck size={16} className="heading-icon text-danger" />
                      Failure Mode &amp; Clinical Risk Prevented
                    </h2>
                    <div className="docs-risk-banner">
                      <AlertTriangle size={18} className="risk-icon" />
                      <div className="risk-content">
                        <div className="risk-badge-label">PREVENTED CLINICAL HAZARD</div>
                        <p>{currentTopic.failureMode}</p>
                      </div>
                    </div>
                  </section>
                )}

                {/* 5. Literature & Standard Benchmark */}
                {currentTopic.literature && (
                  <section className="docs-section-block">
                    <h2 className="docs-section-heading">
                      <FileCode size={16} className="heading-icon text-success" />
                      Academic Literature &amp; Regulatory Reference
                    </h2>
                    <div className="docs-literature-box">
                      <div className="literature-label">PEER-REVIEWED CITATION / STANDARD:</div>
                      <cite className="literature-cite">{currentTopic.literature}</cite>
                    </div>
                  </section>
                )}

                {/* 6. Implementation / Policy Code Snippet */}
                {currentTopic.codeSnippet && (
                  <section className="docs-section-block">
                    <div className="code-heading-row">
                      <h2 className="docs-section-heading">
                        <Terminal size={16} className="heading-icon text-primary" />
                        Reference Implementation &amp; Policy Spec
                      </h2>
                    </div>

                    <div className="docs-code-card">
                      <div className="docs-code-topbar">
                        <div className="code-lang-label">
                          <span className="code-dot red" />
                          <span className="code-dot yellow" />
                          <span className="code-dot green" />
                          <span className="code-filename">
                            {currentTopic.codeLang === 'cedar'
                              ? 'policies/hospital_pacs.cedar'
                              : currentTopic.codeLang === 'json'
                              ? 'schemas/department_entity.json'
                              : 'backend/stress_engine.py'}
                          </span>
                        </div>
                        <button
                          className="docs-code-copy-btn"
                          onClick={() => handleCopyCode(currentTopic.codeSnippet)}
                        >
                          {copiedCode ? (
                            <>
                              <Check size={13} className="text-success" />
                              <span>Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy size={13} />
                              <span>Copy Code</span>
                            </>
                          )}
                        </button>
                      </div>
                      <pre className="docs-code-content">
                        <code>{currentTopic.codeSnippet}</code>
                      </pre>
                    </div>
                  </section>
                )}

                {/* Bottom Pagination: Prev / Next Topic */}
                <footer className="docs-pagination-footer">
                  <div className="pagination-col prev">
                    {prevTopic ? (
                      <button
                        className="pagination-card"
                        onClick={() => handleSelectTopic(prevTopic.id, prevTopic.sectionId)}
                      >
                        <div className="pagination-direction">
                          <ArrowLeft size={13} />
                          <span>Previous Topic</span>
                        </div>
                        <div className="pagination-title">{prevTopic.title}</div>
                      </button>
                    ) : (
                      <div className="pagination-spacer" />
                    )}
                  </div>

                  <div className="pagination-col next">
                    {nextTopic ? (
                      <button
                        className="pagination-card"
                        onClick={() => handleSelectTopic(nextTopic.id, nextTopic.sectionId)}
                      >
                        <div className="pagination-direction">
                          <span>Next Topic</span>
                          <ArrowRight size={13} />
                        </div>
                        <div className="pagination-title">{nextTopic.title}</div>
                      </button>
                    ) : (
                      <div className="pagination-spacer" />
                    )}
                  </div>
                </footer>
              </div>
            </article>
          )}
        </main>
      </div>
    </div>
  );
}
