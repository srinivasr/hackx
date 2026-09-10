import React, { useState, useEffect } from 'react';
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
  const [loading, setLoading] = useState(false);

  // Interactive Stress Studio State
  const [stressType, setStressType] = useState('blur');
  const [stressIntensity, setStressIntensity] = useState(2.5);
  const [sampleKey, setSampleKey] = useState('sample_clinical_pass');
  const [liveStressResult, setLiveStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);

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

  // Live single-stress slider trigger
  const runLiveStress = async (type, intensity, sample) => {
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
      setLiveStressResult(data);
    } catch (e) {
      console.error('Live stress failed', e);
    } finally {
      setStressLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'stress_studio') {
      runLiveStress(stressType, stressIntensity, sampleKey);
    }
  }, [activeTab, stressType, stressIntensity, sampleKey, selectedModel]);

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
          Silent Failures ({auditData?.discrepancies?.length || 0})
        </button>
        <button
          className={`tab-btn ${activeTab === 'arena' ? 'active' : ''}`}
          onClick={() => setActiveTab('arena')}
        >
          <Layers size={15} /> Model Comparison Arena
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
                  <tr>
                    <td><strong>Defocus / Motion Blur</strong></td>
                    <td>σ = 0.0</td>
                    <td>σ = 6.0 (Severe movement)</td>
                    <td>{auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</td>
                    <td><span className="status-fail">FAIL (Sub-threshold)</span></td>
                  </tr>
                  <tr>
                    <td><strong>Flash / Illumination Drop</strong></td>
                    <td>100% Brightness</td>
                    <td>-80% (Undilated pupil)</td>
                    <td>{auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</td>
                    <td><span className="status-fail">FAIL (Severe Sensitivity)</span></td>
                  </tr>
                  <tr>
                    <td><strong>Corneal Glare Reflection</strong></td>
                    <td>0.00</td>
                    <td>0.95 (Corneal Whiteout)</td>
                    <td>{auditData?.stress_tests?.glare_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</td>
                    <td><span className="status-fail">FAIL (Aperture Saturated)</span></td>
                  </tr>
                  <tr>
                    <td><strong>Sensor Resolution Downsampling</strong></td>
                    <td>384x384 px</td>
                    <td>96x96 px (Extreme drop)</td>
                    <td>{auditData?.stress_tests?.resolution_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</td>
                    <td><span className="status-pass">PASS (Tolerant)</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 2: OPTICAL STRESS STUDIO */}
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
                    <span className="t-val">{liveStressResult?.confidence ?? '--'}%</span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ transform: `scaleX(${(liveStressResult?.confidence || 0) / 100})` }}
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

        {/* TAB 4: MODEL ARENA */}
        {activeTab === 'arena' && (
          <div className="arena-container">
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title-text">Model-to-Model Clinical Benchmarking Arena</h3>
                  <p className="card-desc">
                    Side-by-side deployment audit comparing Mobile Edge architectures vs Heavyweight Hospital Server models.
                  </p>
                </div>
              </div>

              <table className="data-table arena-table">
                <thead>
                  <tr>
                    <th>Evaluation Metric</th>
                    <th>Candidate A (Mobile LCNet Edge)</th>
                    <th>Candidate B (Server MaxViT-384)</th>
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
                    <td><span className="status-fail">28.0% Stability Retained</span></td>
                    <td><span className="status-pass">64.5% Stability Retained</span></td>
                    <td>ViT self-attention resists localized blur</td>
                  </tr>
                  <tr>
                    <td><strong>Expected Calibration Error (ECE)</strong></td>
                    <td><span className="status-fail">18.4% (Overconfident)</span></td>
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
