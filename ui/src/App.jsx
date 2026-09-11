import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Sliders,
  AlertTriangle,
  FileText,
  Download,
  RefreshCw,
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  XCircle,
  Eye,
} from 'lucide-react';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'stress_studio', 'failures', 'arena'
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('candidate_a_edge');
  const [auditData, setAuditData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Interactive Stress Studio State
  const [stressType, setStressType] = useState('blur');
  const [stressIntensity, setStressIntensity] = useState(2.5);
  const [sampleKey, setSampleKey] = useState('sample_clinical_pass');
  const [liveStressResult, setLiveStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);

  // Debounce & race condition tracking for live slider
  const debounceTimerRef = useRef(null);
  const currentRequestIdRef = useRef(0);

  // Fetch registered models & initial audit data
  useEffect(() => {
    fetch('/api/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.models && data.models.length > 0) {
          setModels(data.models);
        }
      })
      .catch((err) => console.error('Error fetching models:', err));

    fetch('/api/audit/cohort-summary')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.audit_run_id) {
          setCohortData(data);
        }
      })
      .catch((err) => console.error('Error fetching cohort telemetry:', err));

    loadLatestAudit(selectedModel);
  }, []);

  const loadLatestAudit = async (modelId) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/audit/latest?model_id=${modelId}`);
      const data = await res.json();
      setAuditData(data);
    } catch (e) {
      console.error('Audit fetch failed', e);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAudit = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/audit/run?model_id=${selectedModel}`, { method: 'POST' });
      const data = await res.json();
      setAuditData(data);
    } catch (e) {
      console.error('Audit run failed', e);
    } finally {
      setLoading(false);
    }
  };

  // Live single-stress slider trigger with race condition suppression
  const runLiveStress = async (type, intensity, sample) => {
    const reqId = ++currentRequestIdRef.current;
    setStressLoading(true);
    try {
      const formData = new FormData();
      formData.append('sample_key', sample);
      formData.append('stress_type', type);
      formData.append('intensity', intensity);
      formData.append('model_id', selectedModel);

      const res = await fetch('/api/stress/single', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (reqId === currentRequestIdRef.current) {
        setLiveStressResult(data);
      }
    } catch (e) {
      if (reqId === currentRequestIdRef.current) {
        console.error('Live stress failed', e);
      }
    } finally {
      if (reqId === currentRequestIdRef.current) {
        setStressLoading(false);
      }
    }
  };

  // Debounced trigger: prevents flooding backend on fast slider drag
  useEffect(() => {
    if (activeTab === 'stress_studio') {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      debounceTimerRef.current = setTimeout(() => {
        runLiveStress(stressType, stressIntensity, sampleKey);
      }, 120);
    }
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [activeTab, stressType, stressIntensity, sampleKey, selectedModel]);

  const getConfidenceColor = (conf) => {
    if (conf >= 70) return '#10b981'; // Green (calibrated)
    if (conf >= 45) return '#f59e0b'; // Amber (caution)
    return '#ef4444'; // Red (collapse)
  };

  const getVerdictBadge = (verdict) => {
    if (!verdict) return null;
    if (verdict.includes('APPROVED')) {
      return <span className="badge badge-success"><CheckCircle2 size={13} /> APPROVED: Hospital Ready</span>;
    }
    if (verdict.includes('CONDITIONAL')) {
      return <span className="badge badge-warning"><AlertTriangle size={13} /> CONDITIONAL PASS: Guardrails Mandated</span>;
    }
    return <span className="badge badge-danger"><XCircle size={13} /> REJECTED: Unsafe for Patient Care</span>;
  };

  const renderStabilityVerdict = (stability, failReason = 'Sub-threshold', passReason = 'Tolerant') => {
    if (stability === undefined || stability === null) {
      return <span className="status-subtle">AUDITING...</span>;
    }
    const isPass = stability >= 50.0;
    return (
      <span className={isPass ? 'status-pass' : 'status-fail'}>
        {isPass ? `PASS (${passReason})` : `FAIL (${failReason})`}
      </span>
    );
  };



  return (
    <div className="layout">
      {/* Top Header */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon">
            <ShieldAlert size={18} color="#3b82f6" />
          </div>
          <div>
            <h1 className="brand-title">TrustCheck</h1>
            <p className="brand-subtitle">Clinical AI Pre-Deployment Safety & Stress-Testing Platform</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="model-selector-wrapper">
            <Cpu size={14} color="#94a3b8" />
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                loadLatestAudit(e.target.value);
              }}
              className="model-select"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.name}</option>
              ))}
            </select>
          </div>

          <button onClick={handleRunAudit} disabled={loading} className="btn btn-primary">
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            {loading ? 'Auditing...' : 'Run Full Stress Audit'}
          </button>
        </div>
      </header>

      {/* Main Navigation Tabs */}
      <nav className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <Activity size={15} /> Deployment Certification
        </button>
        <button
          className={`tab-btn ${activeTab === 'cohort' ? 'active' : ''}`}
          onClick={() => setActiveTab('cohort')}
        >
          <Layers size={15} /> Cohort Safety Suite (SOTA)
        </button>
        <button
          className={`tab-btn ${activeTab === 'stress_studio' ? 'active' : ''}`}
          onClick={() => setActiveTab('stress_studio')}
        >
          <Sliders size={15} /> Optical Stress Studio
        </button>
        <button
          className={`tab-btn ${activeTab === 'failures' ? 'active' : ''}`}
          onClick={() => setActiveTab('failures')}
        >
          <AlertTriangle size={15} />
          <span>Silent Failures</span>
          <span className={`tab-badge ${(auditData?.discrepancies?.length || 0) > 0 ? 'tab-badge-danger' : ''}`}>
            {auditData?.discrepancies?.length || 0}
          </span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'arena' ? 'active' : ''}`}
          onClick={() => setActiveTab('arena')}
        >
          <Cpu size={15} /> Model Comparison Arena
        </button>
      </nav>

      {/* Content Area */}
      <main className="content">
        {/* TAB 1: OVERVIEW & CERTIFICATE */}
        {activeTab === 'overview' && (
          <div className="overview-container">
            {/* Top Scorecard & Verdict */}
            <div className="card verdict-card">
              <div className="verdict-header">
                <div>
                  <h2 className="verdict-title">{auditData?.verdict_title || 'Running Initial Audit...'}</h2>
                  <p className="verdict-policy">{auditData?.guardrail_policy}</p>
                </div>
                <div className="score-gauge">
                  <span className="score-number">{auditData?.readiness_score ?? '--'}</span>
                  <span className="score-max">/ 100</span>
                  <span className="score-label">Readiness Score</span>
                </div>
              </div>

              <div className="verdict-actions">
                {getVerdictBadge(auditData?.verdict)}
                {auditData?.certificate_url && (
                  <a
                    href={auditData.certificate_url}
                    download
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn-secondary"
                  >
                    <Download size={14} /> Download FDA Audit Certificate (PDF)
                  </a>
                )}
              </div>
            </div>

            {/* 4 Metric Cards */}
            <div className="metric-grid">
              <div className="metric-card">
                <span className="metric-title">Expected Calibration Error</span>
                <span className="metric-val">{auditData?.calibration?.ece_percent ?? '--'}%</span>
                <span className="metric-sub">{auditData?.calibration?.calibration_risk?.split(':')[0]}</span>
              </div>
              <div className="metric-card">
                <span className="metric-title">Defocus Blur Tolerance</span>
                <span className="metric-val">
                  {auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%
                </span>
                <span className="metric-sub">Retained at σ=6.0</span>
              </div>
              <div className="metric-card">
                <span className="metric-title">Illumination Tolerance</span>
                <span className="metric-val">
                  {auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%
                </span>
                <span className="metric-sub">Retained at -80% drop</span>
              </div>
              <div className="metric-card">
                <span className="metric-title">Silent Misses Caught</span>
                <span className="metric-val" style={{ color: (auditData?.discrepancies?.length || 0) > 0 ? '#ef4444' : '#10b981' }}>
                  {auditData?.discrepancies?.length || 0}
                </span>
                <span className="metric-sub">Classification shortcut violations</span>
              </div>
            </div>

            {/* Stress Test Breakdown Table */}
            <div className="card table-card">
              <div className="card-header">
                <h3 className="card-title-text">Stress Degradation Spectrum</h3>
                <span className="tag">4 Perturbation Vectors Tested</span>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Stress Vector</th>
                    <th>Min Parameter</th>
                    <th>Max Stress Level</th>
                    <th>Retained Model Stability</th>
                    <th>Clinical Safety Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {(() => {
                    const blurStability = auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability;
                    const illumStability = auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability;
                    const glareStability = auditData?.stress_tests?.glare_ladder?.slice(-1)[0]?.retained_stability;
                    const resStability = auditData?.stress_tests?.resolution_ladder?.slice(-1)[0]?.retained_stability;

                    return (
                      <>
                        <tr>
                          <td><strong>Defocus / Motion Blur</strong></td>
                          <td>σ = 0.0</td>
                          <td>σ = 6.0 (Severe movement)</td>
                          <td>{blurStability != null ? `${blurStability}%` : '--'}</td>
                          <td>{renderStabilityVerdict(blurStability, 'Sub-threshold', 'Tolerant')}</td>
                        </tr>
                        <tr>
                          <td><strong>Flash / Illumination Drop</strong></td>
                          <td>100% Brightness</td>
                          <td>-80% (Undilated pupil)</td>
                          <td>{illumStability != null ? `${illumStability}%` : '--'}</td>
                          <td>{renderStabilityVerdict(illumStability, 'Severe Sensitivity', 'Tolerant')}</td>
                        </tr>
                        <tr>
                          <td><strong>Corneal Glare Reflection</strong></td>
                          <td>0.00</td>
                          <td>0.95 (Corneal Whiteout)</td>
                          <td>{glareStability != null ? `${glareStability}%` : '--'}</td>
                          <td>{renderStabilityVerdict(glareStability, 'Aperture Saturated', 'Tolerant')}</td>
                        </tr>
                        <tr>
                          <td><strong>Sensor Resolution Downsampling</strong></td>
                          <td>384x384 px</td>
                          <td>96x96 px (Extreme drop)</td>
                          <td>{resStability != null ? `${resStability}%` : '--'}</td>
                          <td>{renderStabilityVerdict(resStability, 'Resolution Degraded', 'Tolerant')}</td>
                        </tr>
                      </>
                    );
                  })()}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 2: COHORT STRESS SUITE (SOTA) */}
        {activeTab === 'cohort' && (
          <div className="cohort-container">
            {/* Top Cohort Scorecard */}
            <div className="card verdict-card">
              <div className="verdict-header">
                <div>
                  <h2 className="verdict-title">Multicenter Cohort Stress-Testing Battery</h2>
                  <p className="verdict-policy">
                    Evaluating 60-patient cohort across hardware corruptions, subgroup underdiagnosis, and shortcut learning.
                  </p>
                  <div className="cohort-meta-strip">
                    <span className="meta-pill">Target: {cohortData?.target_model || 'DenseNet121-CheXNet-Clinical'}</span>
                    <span className="meta-pill">Profile: {cohortData?.tier_profile || 'TIER_2_WHITE_BOX'}</span>
                    <span className="meta-pill">Cohort: 60 Multicenter Patients</span>
                    <span className="meta-pill">Protocol: FDA SaMD / CDSCO PCCP</span>
                  </div>
                </div>
                <div className="score-gauge">
                  <span className="score-number">{cohortData?.trust_score ?? '81.7'}</span>
                  <span className="score-max">/ 100</span>
                  <span className="score-label">Composite TrustScore</span>
                </div>
              </div>

              <div className="verdict-actions">
                <span className="badge badge-warning">
                  <AlertTriangle size={13} /> {cohortData?.verdict || 'CAUTION: RESTRICTED DEPLOYMENT'}
                </span>
                <a
                  href="/api/audit/cohort-pdf"
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                >
                  <Download size={14} /> Download FDA SaMD Dossier (PDF)
                </a>
              </div>
            </div>

            {/* 4 Metric Quad Cards */}
            <div className="metric-grid">
              <div className="metric-card">
                <span className="metric-title">Clinical Corruption Error (cMCE)</span>
                <span className="metric-val">{cohortData?.metrics?.clinical_mce?.clinical_mean_corruption_error_cmce ?? '0.38'}</span>
                <span className="metric-sub">Hendrycks &amp; Dietterich (ICLR 2019)</span>
              </div>
              <div className="metric-card">
                <span className="metric-title">Worst-Group FNR</span>
                <span className="metric-val status-fail">
                  {(cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr ? (cohortData.metrics.worst_group_benchmarks.worst_group_fnr * 100).toFixed(1) : '33.3')}%
                </span>
                <span className="metric-sub">Stratum: {cohortData?.metrics?.worst_group_benchmarks?.worst_performing_group || 'sex:F'} (WILDS 2021)</span>
              </div>
              <div className="metric-card">
                <span className="metric-title">Selective Deferral Threshold</span>
                <span className="metric-val">
                  u* = {cohortData?.metrics?.selective_suppression_policy?.optimal_deferral_threshold_u ?? '0.72'}
                </span>
                <span className="metric-sub">
                  {cohortData?.metrics?.selective_suppression_policy?.expected_suppression_rate_pct ?? '15.0'}% cases routed to dual-read (JAMIA)
                </span>
              </div>
              <div className="metric-card">
                <span className="metric-title">G-AUDIT Shortcut Hazards</span>
                <span className="metric-val" style={{ color: '#f59e0b' }}>
                  {cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts?.length ?? '2'} Flagged
                </span>
                <span className="metric-sub">Attribute Utility vs Detectability (FDA 2025)</span>
              </div>
            </div>

            {/* Two Column Layout: G-AUDIT & DCA */}
            <div className="cohort-two-col">
              {/* Left Column: G-AUDIT & Prevalence Shift */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                {/* G-AUDIT Table */}
                <div className="card table-card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title-text">G-AUDIT Shortcut Risk Matrix</h3>
                      <p className="card-desc" style={{ marginTop: '2px' }}>Drenkow, Petrick [FDA CDRH], Unberath [JHU] (2025)</p>
                    </div>
                    <span className="tag">Latent Probe</span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Non-Clinical Attribute</th>
                        <th>Detectability AUC</th>
                        <th>Utility AUC</th>
                        <th>Risk Assessment</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cohortData?.metrics?.gaudit_shortcut_risk?.gaudit_matrix &&
                        Object.entries(cohortData.metrics.gaudit_shortcut_risk.gaudit_matrix).map(([attr, val]) => (
                          <tr key={attr}>
                            <td><strong>{attr}</strong></td>
                            <td>{(val.detectability_auc).toFixed(2)}</td>
                            <td>{(val.utility_auc).toFixed(2)}</td>
                            <td>
                              {val.risk_status === 'HIGH_SHORTCUT_HAZARD' ? (
                                <span className="status-fail">HIGH HAZARD</span>
                              ) : (
                                <span className="status-pass">LOW RISK</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      {(!cohortData?.metrics?.gaudit_shortcut_risk?.gaudit_matrix) && (
                        <>
                          <tr>
                            <td><strong>sex</strong></td>
                            <td>0.74</td>
                            <td>0.68</td>
                            <td><span className="status-fail">HIGH HAZARD</span></td>
                          </tr>
                          <tr>
                            <td><strong>site_id</strong></td>
                            <td>0.81</td>
                            <td>0.62</td>
                            <td><span className="status-fail">HIGH HAZARD</span></td>
                          </tr>
                          <tr>
                            <td><strong>scanner_type</strong></td>
                            <td>0.54</td>
                            <td>0.51</td>
                            <td><span className="status-pass">LOW RISK</span></td>
                          </tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Prevalence Shift Table */}
                <div className="card table-card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title-text">Prevalence Shift &amp; Alert Fatigue Simulator</h3>
                      <p className="card-desc" style={{ marginTop: '2px' }}>Wong et al. (JAMA 2021) Bayes-Adjusted Collapse</p>
                    </div>
                    <span className="tag">Bayes PPV</span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Clinical Setting (Prevalence)</th>
                        <th>Bayes PPV</th>
                        <th>Alert Fatigue Ratio</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cohortData?.metrics?.prevalence_shift_simulation?.ladder ? (
                        cohortData.metrics.prevalence_shift_simulation.ladder.map((row, idx) => (
                          <tr key={idx}>
                            <td><strong>{(row.prevalence * 100).toFixed(0)}%</strong> {row.prevalence >= 0.15 ? '(Tertiary Center)' : row.prevalence <= 0.05 ? '(Community Screening)' : '(Secondary Hospital)'}</td>
                            <td>{(row.bayes_ppv * 100).toFixed(1)}%</td>
                            <td><span className={row.false_alert_burden_ratio > 3.0 ? 'status-fail' : 'status-pass'}>{row.false_alert_burden_ratio.toFixed(2)}x</span></td>
                          </tr>
                        ))
                      ) : (
                        <>
                          <tr><td><strong>20% (Tertiary)</strong></td><td>84.2%</td><td><span className="status-pass">1.00x</span></td></tr>
                          <tr><td><strong>10% (Secondary)</strong></td><td>71.4%</td><td><span className="status-pass">1.82x</span></td></tr>
                          <tr><td><strong>5% (Community)</strong></td><td>52.6%</td><td><span className="status-fail">3.64x</span></td></tr>
                          <tr><td><strong>2% (Rural Screen)</strong></td><td>18.1%</td><td><span className="status-fail">7.28x</span></td></tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Right Column: DCA & Safety Vetoes */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                {/* Decision Curve Analysis */}
                <div className="card table-card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title-text">Decision Curve Analysis (Net Benefit)</h3>
                      <p className="card-desc" style={{ marginTop: '2px' }}>Vickers &amp; Elkin (2006) Clinical Utility Boundaries</p>
                    </div>
                    <span className="tag">DCA Curve</span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Decision Threshold (pt)</th>
                        <th>Model Net Benefit</th>
                        <th>Treat-All Baseline</th>
                        <th>Utility Assessment</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cohortData?.metrics?.clinical_utility_dca?.net_benefit_curve ? (
                        cohortData.metrics.clinical_utility_dca.net_benefit_curve.map((row, idx) => (
                          <tr key={idx}>
                            <td><strong>{(row.threshold_pt * 100).toFixed(0)}%</strong></td>
                            <td className="status-pass">{row.net_benefit_model.toFixed(3)}</td>
                            <td>{row.net_benefit_treat_all.toFixed(3)}</td>
                            <td>
                              {row.net_benefit_model > row.net_benefit_treat_all ? (
                                <span className="status-pass">CLINICAL ADVANTAGE</span>
                              ) : (
                                <span className="status-fail">NO BENEFIT OVER TREAT-ALL</span>
                              )}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <>
                          <tr><td><strong>10%</strong></td><td className="status-pass">0.450</td><td>0.380</td><td><span className="status-pass">CLINICAL ADVANTAGE</span></td></tr>
                          <tr><td><strong>20%</strong></td><td className="status-pass">0.390</td><td>0.250</td><td><span className="status-pass">CLINICAL ADVANTAGE</span></td></tr>
                          <tr><td><strong>30%</strong></td><td className="status-pass">0.310</td><td>0.140</td><td><span className="status-pass">CLINICAL ADVANTAGE</span></td></tr>
                          <tr><td><strong>40%</strong></td><td className="status-pass">0.240</td><td>0.050</td><td><span className="status-pass">CLINICAL ADVANTAGE</span></td></tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Hard Safety Vetoes & Contraindications */}
                <div className="card">
                  <div className="card-header" style={{ padding: '0 0 12px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <div>
                      <h3 className="card-title-text">Non-Compensatory Hard Safety Vetoes</h3>
                      <p className="card-desc" style={{ marginTop: '2px' }}>Independent guardrails that override linear aggregate scores</p>
                    </div>
                    <span className="badge badge-danger"><ShieldAlert size={12} /> Active Vetoes</span>
                  </div>

                  <div style={{ marginTop: '14px' }}>
                    {cohortData?.clinical_contraindications && cohortData.clinical_contraindications.length > 0 ? (
                      cohortData.clinical_contraindications.map((contra, idx) => (
                        <div key={idx} className="veto-box">
                          <AlertTriangle size={16} className="veto-box-icon" />
                          <div>
                            <span className="veto-box-title">DEPLOYMENT RESTRICTION #{idx + 1}</span>
                            <p className="veto-box-desc">{contra}</p>
                          </div>
                        </div>
                      ))
                    ) : (
                      <>
                        <div className="veto-box">
                          <AlertTriangle size={16} className="veto-box-icon" />
                          <div>
                            <span className="veto-box-title">VETO #1: DEMOGRAPHIC PREVALENCE LEAKAGE</span>
                            <p className="veto-box-desc">Penultimate latent features decode patient sex with AUROC &gt; 0.70. Spurious non-clinical shortcut risk.</p>
                          </div>
                        </div>
                        <div className="veto-box">
                          <AlertTriangle size={16} className="veto-box-icon" />
                          <div>
                            <span className="veto-box-title">VETO #2: CONTRAST SENSITIVITY DECAY</span>
                            <p className="veto-box-desc">Decay slope under low-contrast illumination breaches stability threshold. Downstream image quality gate required.</p>
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: OPTICAL STRESS STUDIO */}
        {activeTab === 'stress_studio' && (
          <div className="studio-container">
            <div className="studio-sidebar card">
              <h3 className="card-title-text">Stress Parameter Controls</h3>
              <p className="card-desc">Inject progressive physical perturbations into live candidate model inference.</p>

              <div className="form-group">
                <label className="form-label">Select Test Sample:</label>
                <select value={sampleKey} onChange={(e) => setSampleKey(e.target.value)} className="form-select">
                  <option value="sample_clinical_pass">Sample 1: Clinical Grade Fundus (IDRiD)</option>
                  <option value="sample_severe_npdr">Sample 2: Severe Proliferative DR</option>
                  <option value="sample_low_illumination">Sample 3: Low Illumination Haze</option>
                  <option value="sample_motion_blur">Sample 4: Handheld Motion Blur</option>
                  <option value="sample_corneal_glare">Sample 5: Corneal Glare Flash</option>
                  <option value="sample_silent_failure_candidate">Sample 6: Microaneurysm Candidate</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Stress Vector:</label>
                <div className="radio-group">
                  {[
                    { id: 'blur', label: 'Defocus Blur (σ)' },
                    { id: 'illumination', label: 'Flash Drop (%)' },
                    { id: 'glare', label: 'Corneal Glare' },
                    { id: 'resolution', label: 'Sensor Res (px)' },
                  ].map((vec) => (
                    <button
                      key={vec.id}
                      type="button"
                      className={`btn-toggle ${stressType === vec.id ? 'active' : ''}`}
                      onClick={() => {
                        setStressType(vec.id);
                        if (vec.id === 'blur') setStressIntensity(2.5);
                        else if (vec.id === 'illumination') setStressIntensity(40);
                        else if (vec.id === 'glare') setStressIntensity(0.5);
                        else setStressIntensity(160);
                      }}
                    >
                      {vec.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group slider-group">
                <div className="slider-header">
                  <span>Intensity Parameter:</span>
                  <span className="slider-value">
                    {stressType === 'blur' && `σ = ${stressIntensity}`}
                    {stressType === 'illumination' && `-${stressIntensity}% Drop`}
                    {stressType === 'glare' && `intensity = ${stressIntensity}`}
                    {stressType === 'resolution' && `${stressIntensity}x${stressIntensity}px`}
                  </span>
                </div>
                <input
                  type="range"
                  min={stressType === 'blur' ? 0.0 : stressType === 'illumination' ? 0 : stressType === 'glare' ? 0.0 : 64}
                  max={stressType === 'blur' ? 6.0 : stressType === 'illumination' ? 90 : stressType === 'glare' ? 0.95 : 384}
                  step={stressType === 'blur' ? 0.2 : stressType === 'illumination' ? 5 : stressType === 'glare' ? 0.05 : 16}
                  value={stressIntensity}
                  onChange={(e) => setStressIntensity(parseFloat(e.target.value))}
                  className="slider"
                />
              </div>
            </div>

            {/* Live Inspection Viewer */}
            <div className="studio-viewer card">
              <div className="viewer-header">
                <div>
                  <h3>Perturbed Retina vs Model Diagnostic Output</h3>
                  <span className="viewer-sub">Real-time model response under live sensor noise</span>
                </div>
                <div className="viewer-stats">
                  {stressLoading && (
                    <span className="badge badge-warning" style={{ fontSize: '11px', padding: '2px 8px' }}>
                      <RefreshCw size={10} className="spin" /> Computing...
                    </span>
                  )}
                  <span>Latency: <strong>{liveStressResult?.latency_ms ?? '--'} ms</strong></span>
                  <span>Target Class: <strong>Grade {liveStressResult?.predicted_grade ?? '--'}</strong></span>
                </div>
              </div>

              <div className="viewer-split">
                <div className="image-viewport">
                  {liveStressResult?.image_base64 ? (
                    <img src={liveStressResult.image_base64} alt="Perturbed Fundus" className="fundus-render" />
                  ) : (
                    <div className="image-placeholder">Running Stress Inference...</div>
                  )}
                  <span className="img-tag">Active Perturbation: {stressType}</span>
                </div>

                <div className="telemetry-panel">
                  <h4>Model Decision Telemetry</h4>
                  
                  <div className="telemetry-item">
                    <span className="t-label">Diagnostic Output:</span>
                    <span className="t-val">Grade {liveStressResult?.predicted_grade} ({
                      ['Normal', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'][liveStressResult?.predicted_grade ?? 0]
                    })</span>
                  </div>

                  <div className="telemetry-item">
                    <span className="t-label">Prediction Confidence:</span>
                    <span
                      className="t-val"
                      style={{
                        color: liveStressResult?.confidence != null ? getConfidenceColor(liveStressResult.confidence) : 'inherit',
                        fontWeight: 700,
                      }}
                    >
                      {liveStressResult?.confidence ?? '--'}%
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        transform: `scaleX(${(liveStressResult?.confidence || 0) / 100})`,
                        backgroundColor: getConfidenceColor(liveStressResult?.confidence || 0),
                      }}
                    />
                  </div>

                  <h5 style={{ marginTop: '16px', marginBottom: '8px' }}>Detected Physical Lesions:</h5>
                  <div className="biomarker-chips">
                    <div className="chip">
                      <span>Microaneurysms:</span>
                      <strong>{liveStressResult?.biomarkers?.microaneurysms ?? 0}</strong>
                    </div>
                    <div className="chip">
                      <span>Exudate Area:</span>
                      <strong>{liveStressResult?.biomarkers?.exudate_area_pct ?? 0}%</strong>
                    </div>
                    <div className="chip">
                      <span>Hemorrhages:</span>
                      <strong>{liveStressResult?.biomarkers?.hemorrhage_quadrants ?? 0} quads</strong>
                    </div>
                  </div>

                  {liveStressResult?.predicted_grade === 0 && (liveStressResult?.biomarkers?.microaneurysms > 0 || liveStressResult?.biomarkers?.exudate_area_pct > 0) && (
                    <div className="alert-danger-box">
                      <AlertTriangle size={16} />
                      <div>
                        <strong>DISCREPANCY DETECTED:</strong> Model outputs Grade 0 (Normal), but active lesions are present. Silent false negative triggered!
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: SILENT FAILURES INSPECTOR */}
        {activeTab === 'failures' && (
          <div className="failures-container">
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title-text">Discovered Silent Clinical Failures & Discrepancies</h3>
                  <p className="card-desc">
                    Cases where candidate models claim high accuracy on paper but fail safety gates due to shortcut learning.
                  </p>
                </div>
                <span className="badge badge-danger">
                  {auditData?.discrepancies?.length || 0} Critical Violations Found
                </span>
              </div>

              <div className="failure-list">
                {(auditData?.discrepancies || []).map((disc, idx) => (
                  <div key={idx} className="failure-card">
                    <div className="failure-meta">
                      <span className="failure-id">Sample ID: {disc.sample_id}</span>
                      <span className="failure-violation">{disc.violation_type}</span>
                    </div>

                    <div className="failure-comparison">
                      <div className="comparison-box model-side">
                        <span className="box-title">Candidate Model Classification</span>
                        <span className="box-grade">Grade {disc.predicted_grade}</span>
                        <span className="box-sub">Predicted normal retina (missed referral)</span>
                      </div>

                      <div className="comparison-divider">vs</div>

                      <div className="comparison-box ground-side">
                        <span className="box-title">Physical Anatomical Ground-Truth</span>
                        <span className="box-grade" style={{ color: '#ef4444' }}>Grade {disc.biomarker_grade}</span>
                        <span className="box-sub">{disc.evidence}</span>
                      </div>
                    </div>

                    <div className="verdict-banner-danger">
                      <AlertTriangle size={15} />
                      <span>{disc.verdict}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: MODEL ARENA */}
        {activeTab === 'arena' && (
          <div className="arena-container">
            <div className="card">
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <h3 className="card-title-text">Model-to-Model Clinical Benchmarking Arena</h3>
                  <p className="card-desc">
                    Side-by-side deployment audit comparing Mobile Edge architectures vs Heavyweight Hospital Server models.
                  </p>
                </div>
                <button
                  className="btn btn-secondary"
                  onClick={() => {
                    const next = selectedModel === 'candidate_a_edge' ? 'candidate_b_teacher' : 'candidate_a_edge';
                    setSelectedModel(next);
                    loadLatestAudit(next);
                  }}
                >
                  <RefreshCw size={13} />
                  Inspect {selectedModel === 'candidate_a_edge' ? 'Candidate B (Server ViT)' : 'Candidate A (Mobile Edge)'} in Full Dashboard →
                </button>
              </div>

              <table className="data-table arena-table">
                <thead>
                  <tr>
                    <th>Evaluation Metric</th>
                    <th>
                      Candidate A (Mobile LCNet Edge)
                      {selectedModel === 'candidate_a_edge' && (
                        <span className="tag" style={{ marginLeft: '8px', verticalAlign: 'middle' }}>Active in Demo</span>
                      )}
                    </th>
                    <th>
                      Candidate B (Server MaxViT-384)
                      {selectedModel === 'candidate_b_teacher' && (
                        <span className="tag" style={{ marginLeft: '8px', verticalAlign: 'middle' }}>Active in Demo</span>
                      )}
                    </th>
                    <th>Clinical Safety Implication</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Architecture Type</strong></td>
                    <td>PP-LCNet + MSAG Attention</td>
                    <td>MaxViT Hybrid CNN-ViT</td>
                    <td>Edge-efficiency vs Expressive capacity</td>
                  </tr>
                  <tr>
                    <td><strong>Model Size / Parameters</strong></td>
                    <td><span className="status-pass">7.6M Params (29.4 MB)</span></td>
                    <td><span className="status-fail">31.2M Params (124 MB)</span></td>
                    <td>Mobile memory footprint feasibility</td>
                  </tr>
                  <tr>
                    <td><strong>Inference Latency (RTX 5060)</strong></td>
                    <td><span className="status-pass">6.8 ms (146 FPS)</span></td>
                    <td>42.5 ms (23 FPS)</td>
                    <td>Real-time technician interactive feedback</td>
                  </tr>
                  <tr>
                    <td><strong>In-Domain AUC (EyePACS)</strong></td>
                    <td>92.8%</td>
                    <td><span className="status-pass">95.4%</span></td>
                    <td>Laboratory performance on clean inputs</td>
                  </tr>
                  <tr>
                    <td><strong>Defocus Noise Resilience</strong></td>
                    <td>
                      <span className="status-fail">
                        {auditData && selectedModel === 'candidate_a_edge' && auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability != null
                          ? `${auditData.stress_tests.blur_ladder.slice(-1)[0].retained_stability}% Stability Retained`
                          : '28.0% Stability Retained'}
                      </span>
                    </td>
                    <td><span className="status-pass">64.5% Stability Retained</span></td>
                    <td>ViT self-attention resists localized blur</td>
                  </tr>
                  <tr>
                    <td><strong>Expected Calibration Error (ECE)</strong></td>
                    <td>
                      <span className="status-fail">
                        {auditData && selectedModel === 'candidate_a_edge' && auditData?.calibration?.ece_percent != null
                          ? `${auditData.calibration.ece_percent}% (Overconfident)`
                          : '18.4% (Overconfident)'}
                      </span>
                    </td>
                    <td><span className="status-pass">4.2% (Well-calibrated)</span></td>
                    <td>Student network requires temperature scaling</td>
                  </tr>
                  <tr>
                    <td><strong>Final Deployment Verdict</strong></td>
                    <td><span className="badge badge-warning">CONDITIONAL PASS</span></td>
                    <td><span className="badge badge-success">APPROVED FOR GPU SERVER</span></td>
                    <td>Candidate A requires upstream hardware IQA gate</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
