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
  Database,
  Layers,
  Zap,
  CheckCircle2,
  XCircle,
  Eye,
  Upload,
  X,
  Sun,
  Moon,
} from 'lucide-react';
import './App.css';
import AurocCurveViewer from './AurocCurveViewer';

export default function App() {
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('trustcheck-theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('trustcheck-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'cohort', 'stress_studio', 'failures', 'arena'
  const [arenaModality, setArenaModality] = useState('retinal_dr'); // 'retinal_dr', 'chest_xray'
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('dr_lcnet_edge');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('retinal_dr_60');
  const [auditData, setAuditData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cohortLoading, setCohortLoading] = useState(false);

  // Clinical AI Ingestion Center State
  const [showIngestModal, setShowIngestModal] = useState(false);
  const [ingestMode, setIngestMode] = useState('upload_onnx'); // 'upload_onnx', 'register_path', 'upload_dataset'
  const [modelFile, setModelFile] = useState(null);
  const [modelName, setModelName] = useState('');
  const [modelArch, setModelArch] = useState('');
  const [modelTier, setModelTier] = useState('Rural PHC Portable Device');
  const [modelModality, setModelModality] = useState('retinal_fundus');
  const [localPath, setLocalPath] = useState('');
  const [datasetFile, setDatasetFile] = useState(null);
  const [datasetName, setDatasetName] = useState('');
  const [datasetModality, setDatasetModality] = useState('retinal_fundus');
  const [ingestLoading, setIngestLoading] = useState(false);

  // Interactive Stress Studio State
  const [stressType, setStressType] = useState('blur');
  const [stressIntensity, setStressIntensity] = useState(2.5);
  const [sampleKey, setSampleKey] = useState('sample_clinical_pass');
  const [liveStressResult, setLiveStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);

  // Fetch registered models, datasets, & initial audit data
  useEffect(() => {
    fetch('/api/models')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.models && data.models.length > 0) {
          setModels(data.models);
        }
      })
      .catch((err) => console.warn('Models endpoint not reachable yet:', err.message));

    fetch('/api/datasets')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.datasets && data.datasets.length > 0) {
          setDatasets(data.datasets);
        }
      })
      .catch((err) => console.warn('Datasets endpoint not reachable yet:', err.message));

    loadCohortData(selectedModel, selectedDataset);
    loadLatestAudit(selectedModel);
  }, []);

  const isCXR = (mId) => mId === 'cxr_chexnet' || mId === 'cxr_mobilenet_edge';

  const loadCohortData = async (mId, dId) => {
    try {
      const res = await fetch(`/api/audit/cohort-summary?model_id=${mId}&dataset_id=${dId}`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.audit_run_id) {
          setCohortData(data);
        }
      }
    } catch (e) {
      console.warn('Cohort telemetry unavailable:', e.message);
    }
  };

  const handleModelChange = (newModelId) => {
    setSelectedModel(newModelId);
    loadLatestAudit(newModelId);
    // Intelligent auto-pairing with recommended clinical dataset
    let newDatasetId = selectedDataset;
    if (isCXR(newModelId) && selectedDataset !== 'chest_xray_60') {
      newDatasetId = 'chest_xray_60';
      setSelectedDataset('chest_xray_60');
    } else if (!isCXR(newModelId) && selectedDataset === 'chest_xray_60') {
      newDatasetId = 'retinal_dr_60';
      setSelectedDataset('retinal_dr_60');
    }
    loadCohortData(newModelId, newDatasetId);
  };

  const handleDatasetChange = (newDatasetId) => {
    setSelectedDataset(newDatasetId);
    // Intelligent auto-pairing with recommended clinical model
    let newModelId = selectedModel;
    if (newDatasetId === 'chest_xray_60' && !isCXR(selectedModel)) {
      newModelId = 'cxr_chexnet';
      setSelectedModel('cxr_chexnet');
      loadLatestAudit('cxr_chexnet');
    } else if (newDatasetId === 'retinal_dr_60' && isCXR(selectedModel)) {
      newModelId = 'dr_lcnet_edge';
      setSelectedModel('dr_lcnet_edge');
      loadLatestAudit('dr_lcnet_edge');
    }
    loadCohortData(newModelId, newDatasetId);
  };

  const loadLatestAudit = async (modelId) => {
    setLoading(true);
    try {
      const res = await fetch(`/api/audit/latest?model_id=${modelId}`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.audit_id) {
          setAuditData(data);
        }
      }
    } catch (e) {
      console.warn('Audit fetch unavailable:', e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAudit = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/audit/run?model_id=${selectedModel}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setAuditData(data);
        setActiveTab('overview');
      }
    } catch (e) {
      console.warn('Audit run failed:', e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunCohortAudit = async () => {
    setCohortLoading(true);
    try {
      const res = await fetch(`/api/audit/run-cohort?model_id=${selectedModel}&dataset_id=${selectedDataset}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCohortData(data);
      }
    } catch (e) {
      console.warn('Cohort audit run failed:', e.message);
    } finally {
      setCohortLoading(false);
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
      if (res.ok) {
        const data = await res.json();
        setLiveStressResult(data);
      }
    } catch (e) {
      console.warn('Live stress test unavailable:', e.message);
    } finally {
      setStressLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'stress_studio') {
      const timer = setTimeout(() => {
        runLiveStress(stressType, stressIntensity, sampleKey);
      }, 120);
      return () => clearTimeout(timer);
    }
  }, [activeTab, stressType, stressIntensity, sampleKey, selectedModel]);

  const handleUploadModel = async (e) => {
    e.preventDefault();
    if (!modelFile) return;
    setIngestLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', modelFile);
      formData.append('name', modelName || modelFile.name.replace('.onnx', ''));
      formData.append('architecture', modelArch || 'ONNX Clinical Classifier');
      formData.append('target_deployment', modelTier);
      formData.append('modality', modelModality);

      const res = await fetch('/api/models/upload', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        if (data.model) {
          setModels((prev) => [...prev, data.model]);
          setSelectedModel(data.model_id);
          setShowIngestModal(false);
          setModelFile(null);
          setModelName('');
          handleModelChange(data.model_id);
        }
      }
    } catch (err) {
      console.warn('Model upload failed:', err.message);
    } finally {
      setIngestLoading(false);
    }
  };

  const handleRegisterPath = async (e) => {
    e.preventDefault();
    if (!localPath) return;
    setIngestLoading(true);
    try {
      const modelId = `model_${Date.now()}`;
      const formData = new FormData();
      formData.append('model_id', modelId);
      formData.append('name', modelName || localPath.split('/').pop());
      formData.append('path', localPath);
      formData.append('architecture', modelArch || 'ONNX Clinical Classifier');
      formData.append('target_deployment', modelTier);
      formData.append('modality', modelModality);

      const res = await fetch('/api/models/register', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        if (data.model) {
          setModels((prev) => [...prev, data.model]);
          setSelectedModel(modelId);
          setShowIngestModal(false);
          setLocalPath('');
          setModelName('');
          handleModelChange(modelId);
        }
      }
    } catch (err) {
      console.warn('Path registration failed:', err.message);
    } finally {
      setIngestLoading(false);
    }
  };

  const handleUploadDataset = async (e) => {
    e.preventDefault();
    if (!datasetFile) return;
    setIngestLoading(true);
    try {
      const formData = new FormData();
      formData.append('metadata_file', datasetFile);
      formData.append('dataset_name', datasetName || datasetFile.name.replace('.csv', ''));
      formData.append('modality', datasetModality);

      const res = await fetch('/api/datasets/upload', { method: 'POST', body: formData });
      if (res.ok) {
        const data = await res.json();
        if (data.dataset) {
          setDatasets((prev) => [...prev, data.dataset]);
          setSelectedDataset(data.dataset_id);
          setShowIngestModal(false);
          setDatasetFile(null);
          setDatasetName('');
          handleDatasetChange(data.dataset_id);
        }
      }
    } catch (err) {
      console.warn('Dataset upload failed:', err.message);
    } finally {
      setIngestLoading(false);
    }
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

  return (
    <div className="layout">
      {/* Top Header */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon-wrap">
            <ShieldAlert size={16} />
          </div>
          <div className="brand-title-group">
            <h1 className="brand-title">TrustCheck</h1>
            <span className="brand-badge">Clinical SaMD v2.4</span>
          </div>
        </div>

        <div className="header-controls">
          <div className="selector-group">
            <div className="model-selector-wrapper" title="Candidate Model">
              <Cpu size={14} color="#94a3b8" />
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value)}
                className="model-select"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>

            <div className="model-selector-wrapper" title="Evaluation Dataset / Cohort">
              <Database size={14} color="#94a3b8" />
              <select
                value={selectedDataset}
                onChange={(e) => handleDatasetChange(e.target.value)}
                className="model-select"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.sample_count} scans)
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            onClick={() => setShowIngestModal(true)}
            className="btn btn-secondary"
            title="Ingest Custom Model or Cohort"
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <Upload size={14} />
            <span>Ingest</span>
          </button>

          <button
            onClick={toggleTheme}
            className="theme-toggle-btn"
            title={theme === 'dark' ? 'Switch to Light Clinical Workstation Theme' : 'Switch to Dark Obsidian Theme'}
            aria-label="Toggle theme mode"
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>

          <button onClick={handleRunAudit} disabled={loading} className="btn btn-primary">
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            {loading ? 'Auditing...' : 'Run Single Audit'}
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
          Silent Failures ({auditData?.discrepancies?.length || 0})
        </button>
        <button
          className={`tab-btn ${activeTab === 'arena' ? 'active' : ''}`}
          onClick={() => setActiveTab('arena')}
        >
          <Cpu size={15} /> Model Comparison
        </button>
      </nav>

      {/* Content Area */}
      <main className="content">
        {/* TAB 1: OVERVIEW & CERTIFICATE */}
        {activeTab === 'overview' && (
          <div className="overview-container">
            {/* Top Dossier Hero Deck */}
            <div className="dossier-hero">
              <div className="dossier-main">
                <div>
                  <div className="dossier-eyebrow-row">
                    <span className="dossier-kicker">PRE-MARKET EVALUATION</span>
                    <span className="dossier-reg-tag">FDA 21 CFR 820 • CDSCO SaMD CLASS C</span>
                  </div>
                  <h2 className="dossier-headline">{auditData?.verdict_title || 'Running Initial Audit...'}</h2>
                  <p className="dossier-summary">{auditData?.guardrail_policy}</p>
                </div>

                <div className="dossier-action-bar">
                  <div className="dossier-badge-wrap">
                    {getVerdictBadge(auditData?.verdict)}
                  </div>
                  {auditData?.certificate_url && (
                    <a
                      href={auditData.certificate_url}
                      download
                      target="_blank"
                      rel="noreferrer"
                      className="btn btn-secondary"
                    >
                      <Download size={13} /> Download FDA Audit Dossier (PDF)
                    </a>
                  )}
                  <button
                    onClick={() => setActiveTab('failures')}
                    className="btn btn-ghost"
                  >
                    <AlertTriangle size={13} /> Inspect Failures ({auditData?.discrepancies?.length || 0})
                  </button>
                </div>
              </div>

              <div className="dossier-score-card">
                <div>
                  <div className="score-dial-header">
                    <div>
                      <span className="score-number num-tabular">{auditData?.readiness_score ?? '--'}</span>
                      <span className="score-max">/100</span>
                    </div>
                  </div>
                  <div className="score-label-meta">Composite Safety Score</div>
                </div>

                <div className="score-submetrics">
                  <div className="submetric-row">
                    <span>Optical Robustness</span>
                    <span className="num-tabular">{auditData?.optical_robustness ?? '28.0'}%</span>
                  </div>
                  <div className="submetric-track">
                    <div
                      className={`submetric-fill ${(auditData?.optical_robustness ?? 28) < 50 ? 'fail' : (auditData?.optical_robustness ?? 28) < 75 ? 'warn' : 'pass'}`}
                      style={{ width: `${Math.min(100, Math.max(0, auditData?.optical_robustness ?? 28))}%` }}
                    />
                  </div>

                  <div className="submetric-row">
                    <span>Calibration Precision</span>
                    <span className="num-tabular">{auditData?.calibration_precision ?? '14.2'}%</span>
                  </div>
                  <div className="submetric-track">
                    <div
                      className={`submetric-fill ${(auditData?.calibration_precision ?? 14.2) < 50 ? 'fail' : (auditData?.calibration_precision ?? 14.2) < 75 ? 'warn' : 'pass'}`}
                      style={{ width: `${Math.min(100, Math.max(0, auditData?.calibration_precision ?? 14.2))}%` }}
                    />
                  </div>

                  <div className="submetric-row">
                    <span>Clinical Utility Margin</span>
                    <span className="num-tabular">{auditData?.utility_margin ?? '36.5'}%</span>
                  </div>
                  <div className="submetric-track">
                    <div
                      className={`submetric-fill ${(auditData?.utility_margin ?? 36.5) < 50 ? 'fail' : (auditData?.utility_margin ?? 36.5) < 75 ? 'warn' : 'pass'}`}
                      style={{ width: `${Math.min(100, Math.max(0, auditData?.utility_margin ?? 36.5))}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Connected Telemetry Strip */}
            <div className="telemetry-strip">
              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Expected Calibration Error</span>
                  <span className="bay-tag bay-tag-danger">HIGH RISK</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">{auditData?.calibration?.ece_percent ?? '--'}%</span>
                  <span className="bay-unit">ECE</span>
                </div>
                <span className="bay-desc">{auditData?.calibration?.calibration_risk?.split(':')[0] || 'Overconfidence risk'}</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Defocus Blur Tolerance</span>
                  <span className="bay-tag bay-tag-danger">SUB-THRESHOLD</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">
                    {auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%
                  </span>
                  <span className="bay-unit">retained</span>
                </div>
                <span className="bay-desc">Stability drops below 30% at σ=6.0</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Illumination Sensitivity</span>
                  <span className="bay-tag bay-tag-danger">SEVERE</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">
                    {auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%
                  </span>
                  <span className="bay-unit">retained</span>
                </div>
                <span className="bay-desc">Vulnerable at -80% flash drop</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Silent Misses Caught</span>
                  <span className={`bay-tag ${(auditData?.discrepancies?.length || 0) > 0 ? 'bay-tag-danger' : 'bay-tag-success'}`}>
                    {(auditData?.discrepancies?.length || 0) > 0 ? 'CRITICAL' : 'ZERO'}
                  </span>
                </div>
                <div className="bay-value-row">
                  <span
                    className="bay-val num-tabular"
                    style={{
                      color: (auditData?.discrepancies?.length || 0) > 0 ? 'var(--danger)' : 'var(--success)',
                    }}
                  >
                    {auditData?.discrepancies?.length || 0}
                  </span>
                  <span className="bay-unit">patients</span>
                </div>
                <span className="bay-desc">Active physical lesions classified Normal</span>
              </div>
            </div>

            {/* Stress Test Breakdown Table */}
            <div className="card table-card">
              <div className="card-header">
                <h3 className="card-title-text">Stress Degradation Spectrum</h3>
                <span className="tag">4 Perturbation Vectors Tested</span>
              </div>
              <div className="table-wrapper">
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
                      <td className="num-tabular">σ = 0.0</td>
                      <td className="num-tabular">σ = 6.0 (Severe movement)</td>
                      <td>
                        <div className="table-retention-cell">
                          <span className="num-tabular">{auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</span>
                          <div className="retention-mini-track">
                            <div
                              className="retention-mini-fill fail"
                              style={{ width: `${Math.min(100, auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability || 0)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td><span className="status-pill status-pill-fail">FAIL • Sub-threshold</span></td>
                    </tr>
                    <tr>
                      <td><strong>Flash / Illumination Drop</strong></td>
                      <td className="num-tabular">100% Brightness</td>
                      <td className="num-tabular">-80% (Undilated pupil)</td>
                      <td>
                        <div className="table-retention-cell">
                          <span className="num-tabular">{auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</span>
                          <div className="retention-mini-track">
                            <div
                              className="retention-mini-fill fail"
                              style={{ width: `${Math.min(100, auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability || 0)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td><span className="status-pill status-pill-fail">FAIL • Severe Sensitivity</span></td>
                    </tr>
                    <tr>
                      <td><strong>Corneal Glare Reflection</strong></td>
                      <td className="num-tabular">0.00</td>
                      <td className="num-tabular">0.95 (Corneal Whiteout)</td>
                      <td>
                        <div className="table-retention-cell">
                          <span className="num-tabular">{auditData?.stress_tests?.glare_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</span>
                          <div className="retention-mini-track">
                            <div
                              className="retention-mini-fill fail"
                              style={{ width: `${Math.min(100, auditData?.stress_tests?.glare_ladder?.slice(-1)[0]?.retained_stability || 0)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td><span className="status-pill status-pill-fail">FAIL • Aperture Saturated</span></td>
                    </tr>
                    <tr>
                      <td><strong>Sensor Resolution Downsampling</strong></td>
                      <td className="num-tabular">384×384 px</td>
                      <td className="num-tabular">96×96 px (Extreme drop)</td>
                      <td>
                        <div className="table-retention-cell">
                          <span className="num-tabular">{auditData?.stress_tests?.resolution_ladder?.slice(-1)[0]?.retained_stability ?? '--'}%</span>
                          <div className="retention-mini-track">
                            <div
                              className="retention-mini-fill pass"
                              style={{ width: `${Math.min(100, auditData?.stress_tests?.resolution_ladder?.slice(-1)[0]?.retained_stability || 0)}%` }}
                            />
                          </div>
                        </div>
                      </td>
                      <td><span className="status-pill status-pill-pass">PASS • Tolerant</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: COHORT STRESS SUITE (SOTA) */}
        {activeTab === 'cohort' && (
          <div className="cohort-container">
            {/* Top Cohort Dossier Hero Deck */}
            <div className="dossier-hero">
              <div className="dossier-main">
                <div>
                  <div className="dossier-eyebrow-row">
                    <span className="dossier-kicker">MULTICENTER STRESS BATTERY</span>
                    <span className="dossier-reg-tag">FDA SaMD / CDSCO PCCP PROTOCOL</span>
                  </div>
                  <h2 className="dossier-headline">Multicenter Cohort Stress-Testing Battery</h2>
                  <p className="dossier-summary">
                    Evaluating 60-patient cohort across hardware corruptions, subgroup underdiagnosis, and shortcut learning.
                  </p>
                  <div className="cohort-meta-strip">
                    <span className="meta-pill">Target: {cohortData?.target_model || selectedModel}</span>
                    <span className="meta-pill">Modality: {cohortData?.modality ? cohortData.modality.toUpperCase() : 'CLINICAL'}</span>
                    <span className="meta-pill">Cohort: {cohortData?.dataset_metadata?.name || selectedDataset}</span>
                    <span className="meta-pill">Protocol: FDA SaMD / CDSCO PCCP</span>
                  </div>
                </div>

                <div className="dossier-action-bar">
                  <span className="badge badge-warning">
                    <AlertTriangle size={13} /> {cohortData?.verdict || 'CAUTION: RESTRICTED DEPLOYMENT'}
                  </span>
                  <button
                    onClick={handleRunCohortAudit}
                    disabled={cohortLoading}
                    className="btn btn-primary"
                  >
                    <RefreshCw size={14} className={cohortLoading ? 'spin' : ''} />
                    {cohortLoading ? 'Auditing Multicenter Cohort...' : 'Run Cohort Audit Battery'}
                  </button>
                  <a
                    href={`/api/audit/cohort-pdf?model_id=${selectedModel}&dataset_id=${selectedDataset}`}
                    download
                    target="_blank"
                    rel="noreferrer"
                    className="btn btn-secondary"
                  >
                    <Download size={14} /> Download FDA SaMD Dossier (PDF)
                  </a>
                </div>
              </div>

              <div className="dossier-score-card">
                <div>
                  <div className="score-dial-header">
                    <div>
                      <span className="score-number num-tabular">{cohortData?.trust_score ?? '--'}</span>
                      <span className="score-max">/100</span>
                    </div>
                  </div>
                  <div className="score-label-meta">Composite TrustScore</div>
                </div>

                <div className="score-submetrics">
                  <div className="submetric-row">
                    <span>Clinical Corruption Robustness</span>
                    <span className="num-tabular">62.0%</span>
                  </div>
                  <div className="submetric-track">
                    <div className="submetric-fill warn" style={{ width: '62%' }} />
                  </div>

                  <div className="submetric-row">
                    <span>Subgroup Equity Parity</span>
                    <span className="num-tabular">66.7%</span>
                  </div>
                  <div className="submetric-track">
                    <div className="submetric-fill fail" style={{ width: '66.7%' }} />
                  </div>

                  <div className="submetric-row">
                    <span>Safe Deferral Yield</span>
                    <span className="num-tabular">85.0%</span>
                  </div>
                  <div className="submetric-track">
                    <div className="submetric-fill pass" style={{ width: '85%' }} />
                  </div>
                </div>
              </div>
            </div>

            {/* 4 Connected Telemetry Bays */}
            <div className="telemetry-strip">
              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Clinical Corruption Error</span>
                  <span className="bay-tag bay-tag-warning">cMCE</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">{cohortData?.metrics?.clinical_mce?.clinical_mean_corruption_error_cmce ?? '0.38'}</span>
                  <span className="bay-unit">index</span>
                </div>
                <span className="bay-desc">Hendrycks &amp; Dietterich (ICLR 2019)</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Worst-Group FNR</span>
                  <span className="bay-tag bay-tag-danger">HIGH RISK</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular" style={{ color: 'var(--danger)' }}>
                    {(cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr ? (cohortData.metrics.worst_group_benchmarks.worst_group_fnr * 100).toFixed(1) : '33.3')}%
                  </span>
                  <span className="bay-unit">FNR</span>
                </div>
                <span className="bay-desc">Stratum: {cohortData?.metrics?.worst_group_benchmarks?.worst_performing_group || 'sex:F'} (WILDS 2021)</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Selective Deferral</span>
                  <span className="bay-tag bay-tag-success">OPTIMAL</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">
                    u* = {cohortData?.metrics?.selective_suppression_policy?.optimal_deferral_threshold_u ?? '0.72'}
                  </span>
                </div>
                <span className="bay-desc">
                  {cohortData?.metrics?.selective_suppression_policy?.expected_suppression_rate_pct ?? '15.0'}% cases routed to dual-read (JAMIA)
                </span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">G-AUDIT Hazards</span>
                  <span className="bay-tag bay-tag-warning">
                    {cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts?.length ?? '2'} FLAGGED
                  </span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular" style={{ color: 'var(--warning)' }}>
                    {cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts?.length ?? '2'} Shortcuts
                  </span>
                </div>
                <span className="bay-desc">Attribute Utility vs Detectability (FDA 2025)</span>
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
                  <div className="table-wrapper">
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
                                {(val.status === 'SHORTCUT_HAZARD' || val.risk_status === 'HIGH_SHORTCUT_HAZARD') ? (
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
                  <div className="table-wrapper">
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
                  <div className="table-wrapper">
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
                        <span className="box-grade" style={{ color: 'var(--danger)' }}>Grade {disc.biomarker_grade}</span>
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
              <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <h3 className="card-title-text">Model-to-Model Clinical Benchmarking</h3>
                  <p className="card-desc">
                    Side-by-side deployment audit comparing Mobile Edge architectures vs Heavyweight Hospital Server models.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className={`btn ${arenaModality === 'retinal_dr' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setArenaModality('retinal_dr')}
                    style={{ fontSize: '13px', padding: '6px 12px' }}
                  >
                    Diabetic Retinopathy (Fundus)
                  </button>
                  <button
                    className={`btn ${arenaModality === 'chest_xray' ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setArenaModality('chest_xray')}
                    style={{ fontSize: '13px', padding: '6px 12px' }}
                  >
                    Chest Radiography (CXR)
                  </button>
                </div>
              </div>

              {/* Interactive AUROC Diagnostic Discrimination Deck */}
              <AurocCurveViewer modality={arenaModality} />

              {arenaModality === 'retinal_dr' ? (
                <div className="table-wrapper">
                  <table className="data-table arena-table">
                    <thead>
                      <tr>
                        <th>Evaluation Metric</th>
                        <th>DR Mobile LCNet (Edge)</th>
                        <th>DR ResNet Teacher (Server)</th>
                        <th>Clinical Safety Implication</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Architecture Type</strong></td>
                        <td>PP-LCNet + MSAG Attention</td>
                        <td>ResNet18 Deep Ensemble</td>
                        <td>Edge-efficiency vs Expressive capacity</td>
                      </tr>
                      <tr>
                        <td><strong>Model Size / Parameters</strong></td>
                        <td><span className="status-pass">7.6M Params (12.7 MB)</span></td>
                        <td><span className="status-fail">11.2M Params (44.6 MB)</span></td>
                        <td>Mobile memory footprint feasibility</td>
                      </tr>
                      <tr>
                        <td><strong>Inference Latency (RTX 5060)</strong></td>
                        <td><span className="status-pass">6.8 ms (146 FPS)</span></td>
                        <td>14.2 ms (70 FPS)</td>
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
                        <td>Teacher representation resists localized blur</td>
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
                        <td>DR Mobile LCNet requires upstream hardware IQA gate</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="table-wrapper">
                  <table className="data-table arena-table">
                    <thead>
                      <tr>
                        <th>Evaluation Metric</th>
                        <th>CXR MobileNetV2 (Bedside Cart Edge)</th>
                        <th>CXR CheXNet DenseNet121 (Hospital Grade)</th>
                        <th>Clinical Safety Implication</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td><strong>Architecture Type</strong></td>
                        <td>MobileNetV2 Depthwise-Conv</td>
                        <td>DenseNet-121 Feature Reuse</td>
                        <td>Portable ICU Cart vs Radiology Workstation</td>
                      </tr>
                      <tr>
                        <td><strong>Model Size / Parameters</strong></td>
                        <td><span className="status-pass">3.5M Params (8.4 MB)</span></td>
                        <td><span className="status-fail">7.0M Params (27.7 MB)</span></td>
                        <td>Battery runtime & low thermal envelope</td>
                      </tr>
                      <tr>
                        <td><strong>Inference Latency (RTX 5060)</strong></td>
                        <td><span className="status-pass">4.1 ms (240 FPS)</span></td>
                        <td>12.8 ms (78 FPS)</td>
                        <td>Instantaneous bedside triage at patient bed</td>
                      </tr>
                      <tr>
                        <td><strong>In-Domain AUC (NIH CXR-14)</strong></td>
                        <td>74.2%</td>
                        <td><span className="status-pass">86.8%</span></td>
                        <td>Baseline consolidation/infiltrate detection</td>
                      </tr>
                      <tr>
                        <td><strong>CR vs DR Contrast Sensitivity</strong></td>
                        <td><span className="status-fail">42.1% Stability Retained</span></td>
                        <td><span className="status-pass">78.4% Stability Retained</span></td>
                        <td>Edge model fails when contrast drops &gt; 15%</td>
                      </tr>
                      <tr>
                        <td><strong>Expected Calibration Error (ECE)</strong></td>
                        <td><span className="status-fail">10.8% (Borderline Overconfident)</span></td>
                        <td><span className="status-pass">3.8% (Calibrated Posterior)</span></td>
                        <td>Overconfidence on ambiguous lung opacities</td>
                      </tr>
                      <tr>
                        <td><strong>Final Deployment Verdict</strong></td>
                        <td><span className="badge badge-danger">RESTRICTED / CAUTION</span></td>
                        <td><span className="badge badge-success">APPROVED FOR WORKSTATION</span></td>
                        <td>MobileNet requires mandatory radiologist over-read</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* CLINICAL AI INGESTION CENTER MODAL */}
      {showIngestModal && (
        <div className="modal-backdrop" onClick={() => setShowIngestModal(false)}>
          <div className="modal-window card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 className="modal-title">Clinical AI Ingestion Center</h3>
                <p className="modal-desc">
                  Onboard external ONNX neural networks and multicenter clinical cohorts into TrustCheck.
                </p>
              </div>
              <button onClick={() => setShowIngestModal(false)} className="btn-icon">
                <X size={18} />
              </button>
            </div>

            <div className="modal-tabs">
              <button
                type="button"
                className={`modal-tab-btn ${ingestMode === 'upload_onnx' ? 'active' : ''}`}
                onClick={() => setIngestMode('upload_onnx')}
              >
                <Upload size={14} /> Upload .ONNX Model
              </button>
              <button
                type="button"
                className={`modal-tab-btn ${ingestMode === 'register_path' ? 'active' : ''}`}
                onClick={() => setIngestMode('register_path')}
              >
                <Cpu size={14} /> Register Local Path
              </button>
              <button
                type="button"
                className={`modal-tab-btn ${ingestMode === 'upload_dataset' ? 'active' : ''}`}
                onClick={() => setIngestMode('upload_dataset')}
              >
                <Database size={14} /> Upload Cohort CSV
              </button>
            </div>

            {/* TAB 1: UPLOAD ONNX */}
            {ingestMode === 'upload_onnx' && (
              <form onSubmit={handleUploadModel} className="modal-form">
                <div className="form-group">
                  <label className="form-label">Select .ONNX Model File from Local Storage:</label>
                  <div className="file-dropzone">
                    <input
                      type="file"
                      accept=".onnx"
                      onChange={(e) => setModelFile(e.target.files[0])}
                      className="file-input"
                      id="onnx-file-input"
                    />
                    <label htmlFor="onnx-file-input" className="file-dropzone-label">
                      <Upload size={24} color="#3b82f6" />
                      <span>{modelFile ? modelFile.name : 'Click to browse local .onnx weights'}</span>
                      <small style={{ color: 'var(--text-muted)' }}>Supports FP32/FP16 exported ONNX graphs</small>
                    </label>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Model Display Name:</label>
                    <input
                      type="text"
                      placeholder="e.g. ResNet-50 Hospital Clinical Classifier"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="form-input"
                    />
                  </div>
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Clinical Modality:</label>
                    <select
                      value={modelModality}
                      onChange={(e) => setModelModality(e.target.value)}
                      className="form-select"
                    >
                      <option value="retinal_fundus">Ophthalmology (Retinal Fundus)</option>
                      <option value="chest_xray">Pulmonology (Chest Radiography)</option>
                    </select>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Architecture Profile:</label>
                    <input
                      type="text"
                      placeholder="e.g. ResNet50 + SE Attention (25.6M Params)"
                      value={modelArch}
                      onChange={(e) => setModelArch(e.target.value)}
                      className="form-input"
                    />
                  </div>
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Deployment Environment Tier:</label>
                    <select
                      value={modelTier}
                      onChange={(e) => setModelTier(e.target.value)}
                      className="form-select"
                    >
                      <option value="Rural PHC Portable Device">Tier 1: Rural PHC Smartphone / Edge</option>
                      <option value="District Hospital GPU Server">Tier 2: District Hospital Server</option>
                      <option value="Tertiary PACS Sandbox">Tier 3: Tertiary Cloud PACS</option>
                    </select>
                  </div>
                </div>

                <div className="modal-actions">
                  <button
                    type="button"
                    onClick={() => setShowIngestModal(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={ingestLoading || !modelFile}
                    className="btn btn-primary"
                  >
                    <RefreshCw size={14} className={ingestLoading ? 'spin' : ''} />
                    {ingestLoading ? 'Ingesting Model...' : 'Ingest Model & Register'}
                  </button>
                </div>
              </form>
            )}

            {/* TAB 2: REGISTER HARDWARE PATH */}
            {ingestMode === 'register_path' && (
              <form onSubmit={handleRegisterPath} className="modal-form">
                <div className="form-group">
                  <label className="form-label">Absolute or Relative Local File Path:</label>
                  <input
                    type="text"
                    placeholder="e.g. assets/models/dr_retinal_resnet_teacher.onnx"
                    value={localPath}
                    onChange={(e) => setLocalPath(e.target.value)}
                    className="form-input"
                    required
                  />
                  <small style={{ color: 'var(--text-muted)', marginTop: '4px', display: 'block' }}>
                    Path on the host machine running TrustCheck.
                  </small>
                </div>

                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Model Display Name:</label>
                    <input
                      type="text"
                      placeholder="e.g. Custom Model Path"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                      className="form-input"
                    />
                  </div>
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Clinical Modality:</label>
                    <select
                      value={modelModality}
                      onChange={(e) => setModelModality(e.target.value)}
                      className="form-select"
                    >
                      <option value="retinal_fundus">Ophthalmology (Retinal Fundus)</option>
                      <option value="chest_xray">Pulmonology (Chest Radiography)</option>
                    </select>
                  </div>
                </div>

                <div className="modal-actions">
                  <button
                    type="button"
                    onClick={() => setShowIngestModal(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={ingestLoading || !localPath}
                    className="btn btn-primary"
                  >
                    <RefreshCw size={14} className={ingestLoading ? 'spin' : ''} />
                    {ingestLoading ? 'Registering...' : 'Register Local Model'}
                  </button>
                </div>
              </form>
            )}

            {/* TAB 3: UPLOAD COHORT CSV */}
            {ingestMode === 'upload_dataset' && (
              <form onSubmit={handleUploadDataset} className="modal-form">
                <div className="form-group">
                  <label className="form-label">Select Cohort Metadata (.CSV) File:</label>
                  <div className="file-dropzone">
                    <input
                      type="file"
                      accept=".csv"
                      onChange={(e) => setDatasetFile(e.target.files[0])}
                      className="file-input"
                      id="csv-file-input"
                    />
                    <label htmlFor="csv-file-input" className="file-dropzone-label">
                      <FileText size={24} color="#3b82f6" />
                      <span>{datasetFile ? datasetFile.name : 'Click to select cohort metadata (.csv)'}</span>
                      <small style={{ color: 'var(--text-muted)' }}>Must include patient_id, image_path, ground_truth</small>
                    </label>
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Cohort Dataset Name:</label>
                    <input
                      type="text"
                      placeholder="e.g. AIIMS Delhi Retinal Cohort"
                      value={datasetName}
                      onChange={(e) => setDatasetName(e.target.value)}
                      className="form-input"
                    />
                  </div>
                  <div className="form-group" style={{ flex: 1 }}>
                    <label className="form-label">Clinical Modality:</label>
                    <select
                      value={datasetModality}
                      onChange={(e) => setDatasetModality(e.target.value)}
                      className="form-select"
                    >
                      <option value="retinal_fundus">Ophthalmology (Retinal Fundus)</option>
                      <option value="chest_xray">Pulmonology (Chest Radiography)</option>
                    </select>
                  </div>
                </div>

                <div className="modal-actions">
                  <button
                    type="button"
                    onClick={() => setShowIngestModal(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={ingestLoading || !datasetFile}
                    className="btn btn-primary"
                  >
                    <RefreshCw size={14} className={ingestLoading ? 'spin' : ''} />
                    {ingestLoading ? 'Ingesting Dataset...' : 'Ingest Cohort Dataset'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
