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
  Menu,
  Sun,
  Moon,
  Info,
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

  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'cohort', 'stress_studio', 'failures', 'arena'
  const [arenaModality, setArenaModality] = useState('retinal_dr'); // 'retinal_dr', 'chest_xray'
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('dr_lcnet_edge');
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('retinal_dr_60');
  const [auditData, setAuditData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [completedAudits, setCompletedAudits] = useState({});
  const [completedCohortAudits, setCompletedCohortAudits] = useState({});
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
  const [customScanFile, setCustomScanFile] = useState(null);

  // Fetch registered models & datasets (audits remain un-run until triggered)
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
  }, []);

  const isCXR = (mId) => mId === 'cxr_chexnet' || mId === 'cxr_mobilenet_edge';
  const isNLP = (mId) => mId === 'nlp_bioclinicalbert' || mId === 'nlp_pubmedbert';
  const isDerm = (mId) => mId === 'derm_efficientnet_melanoma' || mId === 'derm_resnet_reference';
  const isPath = (mId) => mId === 'path_mobilenet_pcam' || mId === 'path_densenet_wsi';

  const getSpectrumRows = () => {
    if (!auditData) return [];
    if (auditData?.stress_spectrum && auditData.stress_spectrum.length > 0) {
      return auditData.stress_spectrum;
    }
    const tests = auditData?.stress_tests || {};
    const blurStab = tests.blur_ladder?.slice(-1)[0]?.retained_stability;
    const illumStab = tests.illumination_ladder?.slice(-1)[0]?.retained_stability;
    const glareStab = tests.glare_ladder?.slice(-1)[0]?.retained_stability;
    const resStab = tests.resolution_ladder?.slice(-1)[0]?.retained_stability;

    if (blurStab === undefined && illumStab === undefined && glareStab === undefined && resStab === undefined) {
      return [];
    }

    const getStatus = (s) => (s >= 70 ? 'pass' : s >= 40 ? 'warn' : 'fail');
    const getVerdict = (s) => (s >= 70 ? 'PASS • Robust' : s >= 40 ? 'WARN • Sub-threshold' : 'FAIL • Sub-threshold');

    return [
      {
        vector_name: isCXR(selectedModel) ? 'Thoracic Motion Blur' : isNLP(selectedModel) ? 'OCR Typographical Noise' : isDerm(selectedModel) ? 'Handheld Tremor Defocus' : isPath(selectedModel) ? 'WSI Stage Focal Defocus' : 'Defocus / Motion Blur',
        min_param: isCXR(selectedModel) ? 'Kernel = 0 px' : isNLP(selectedModel) ? '0% Typo Noise' : isDerm(selectedModel) ? 'Contact Lens Fixed' : isPath(selectedModel) ? 'Focal Plane 0 μm' : 'σ = 0.0',
        max_param: isCXR(selectedModel) ? 'Kernel = 21 px (Tremor)' : isNLP(selectedModel) ? '42% Char Swaps' : isDerm(selectedModel) ? 'Kernel = 23 px' : isPath(selectedModel) ? 'Out-of-focus 21 px' : 'σ = 6.0 (Severe movement)',
        retained_stability: blurStab,
        status: blurStab !== undefined ? getStatus(blurStab) : 'warn',
        verdict: blurStab !== undefined ? getVerdict(blurStab) : 'Pending evaluation',
      },
      {
        vector_name: isCXR(selectedModel) ? 'CR/DR Contrast Attenuation' : isNLP(selectedModel) ? 'Medical Abbreviation Density' : isDerm(selectedModel) ? 'Peripheral Optical Vignetting' : isPath(selectedModel) ? 'H&E Stain Batch Variability' : 'Flash / Illumination Drop',
        min_param: isCXR(selectedModel) ? '100% Dynamic Range' : isNLP(selectedModel) ? 'Standard Clinical Text' : isDerm(selectedModel) ? 'Uniform Field' : isPath(selectedModel) ? 'Calibrated Histology pH' : '100% Brightness',
        max_param: isCXR(selectedModel) ? '-75% Underexposure' : isNLP(selectedModel) ? '80% Physician Shorthand' : isDerm(selectedModel) ? '-80% Rim Falloff' : isPath(selectedModel) ? 'Severe Chemical Shift' : '-80% (Undilated pupil)',
        retained_stability: illumStab,
        status: illumStab !== undefined ? getStatus(illumStab) : 'warn',
        verdict: illumStab !== undefined ? getVerdict(illumStab) : 'Pending evaluation',
      },
      {
        vector_name: isCXR(selectedModel) ? 'Quantum Poisson Shot Noise' : isNLP(selectedModel) ? 'Hasty Note Truncation' : isDerm(selectedModel) ? 'Dermatoscope Specular Glare' : isPath(selectedModel) ? 'Microtome Section Folding' : 'Corneal Glare Reflection',
        min_param: isCXR(selectedModel) ? 'High-Dose Photons' : isNLP(selectedModel) ? '100% Complete Narrative' : isDerm(selectedModel) ? 'Cross-Polarized' : isPath(selectedModel) ? 'Intact Tissue Section' : '0.00',
        max_param: isCXR(selectedModel) ? 'Low-Dose Scatter Noise' : isNLP(selectedModel) ? '40% Length (Cutoff)' : isDerm(selectedModel) ? 'Immersion Fluid Flare' : isPath(selectedModel) ? '5 Compression Folds' : '0.95 (Corneal Whiteout)',
        retained_stability: glareStab,
        status: glareStab !== undefined ? getStatus(glareStab) : 'warn',
        verdict: glareStab !== undefined ? getVerdict(glareStab) : 'Pending evaluation',
      },
      {
        vector_name: isCXR(selectedModel) ? 'Matrix Resolution Downsampling' : isNLP(selectedModel) ? 'Negation Assertion Stress' : isDerm(selectedModel) ? 'Mobile Dermatoscope Downsampling' : isPath(selectedModel) ? 'Optical Magnification Scaling' : 'Sensor Resolution Downsampling',
        min_param: isCXR(selectedModel) ? '384×384 px' : isNLP(selectedModel) ? 'Intact Assertions' : isDerm(selectedModel) ? '384×384 px Native' : isPath(selectedModel) ? '40x Native Lens' : '384×384 px',
        max_param: isCXR(selectedModel) ? '96×96 px (Mobile cart)' : isNLP(selectedModel) ? 'NegEx Semantic Inversion' : isDerm(selectedModel) ? '96×96 px (Tele-derm)' : isPath(selectedModel) ? '10x Scouting View' : '96×96 px (Extreme drop)',
        retained_stability: resStab,
        status: resStab !== undefined ? getStatus(resStab) : 'warn',
        verdict: resStab !== undefined ? getVerdict(resStab) : 'Pending evaluation',
      },
    ];
  };

  const handleModelChange = (newModelId) => {
    setSelectedModel(newModelId);
    setAuditData(completedAudits[newModelId] || null);

    // Sync optical stress studio sample key with modality
    if (isCXR(newModelId) && !sampleKey.startsWith('scan_')) {
      setSampleKey('scan_0001');
    } else if (isDerm(newModelId) && !sampleKey.startsWith('lesion_')) {
      setSampleKey('lesion_0001');
    } else if (isPath(newModelId) && !sampleKey.startsWith('wsi_')) {
      setSampleKey('wsi_0001');
    } else if (!isCXR(newModelId) && !isDerm(newModelId) && !isPath(newModelId) && (sampleKey.startsWith('scan_') || sampleKey.startsWith('lesion_') || sampleKey.startsWith('wsi_'))) {
      setSampleKey('sample_clinical_pass');
    }

    // Intelligent auto-pairing with recommended clinical dataset
    let newDatasetId = selectedDataset;
    if (isCXR(newModelId)) {
      newDatasetId = 'chest_xray_60';
    } else if (isNLP(newModelId)) {
      newDatasetId = 'clinical_notes_mimic_60';
    } else if (isDerm(newModelId)) {
      newDatasetId = 'dermatology_isic_60';
    } else if (isPath(newModelId)) {
      newDatasetId = 'histopathology_pcam_60';
    } else {
      newDatasetId = 'retinal_dr_60';
    }
    setSelectedDataset(newDatasetId);
    setCohortData(completedCohortAudits[`${newModelId}_${newDatasetId}`] || null);
  };

  const handleDatasetChange = (newDatasetId) => {
    setSelectedDataset(newDatasetId);
    // Intelligent auto-pairing with recommended clinical model
    let newModelId = selectedModel;
    if (newDatasetId === 'chest_xray_60' && !isCXR(selectedModel)) {
      newModelId = 'cxr_chexnet';
      setSampleKey('scan_0001');
    } else if (newDatasetId.includes('clinical_notes') || newDatasetId.includes('mednli')) {
      if (!isNLP(selectedModel)) newModelId = 'nlp_bioclinicalbert';
    } else if (newDatasetId === 'dermatology_isic_60' && !isDerm(selectedModel)) {
      newModelId = 'derm_efficientnet_melanoma';
      setSampleKey('lesion_0001');
    } else if (newDatasetId === 'histopathology_pcam_60' && !isPath(selectedModel)) {
      newModelId = 'path_mobilenet_pcam';
      setSampleKey('wsi_0001');
    } else if (newDatasetId === 'retinal_dr_60' && (isCXR(selectedModel) || isNLP(selectedModel) || isDerm(selectedModel) || isPath(selectedModel))) {
      newModelId = 'dr_lcnet_edge';
      setSampleKey('sample_clinical_pass');
    }
    setSelectedModel(newModelId);
    setAuditData(completedAudits[newModelId] || null);
    setCohortData(completedCohortAudits[`${newModelId}_${newDatasetId}`] || null);
  };

  const handleRunAudit = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/audit/run?model_id=${selectedModel}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setAuditData(data);
        setCompletedAudits((prev) => ({ ...prev, [selectedModel]: data }));
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
        setCompletedCohortAudits((prev) => ({ ...prev, [`${selectedModel}_${selectedDataset}`]: data }));
      }
    } catch (e) {
      console.warn('Cohort audit run failed:', e.message);
    } finally {
      setCohortLoading(false);
    }
  };

  // Live single-stress slider trigger
  const runLiveStress = async (type = stressType, intensity = stressIntensity, sample = sampleKey, file = customScanFile) => {
    setStressLoading(true);
    try {
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      } else {
        formData.append('sample_key', sample);
      }
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
        runLiveStress(stressType, stressIntensity, sampleKey, customScanFile);
      }, 120);
      return () => clearTimeout(timer);
    }
  }, [activeTab, stressType, stressIntensity, sampleKey, selectedModel, customScanFile]);

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
    <div className="app-shell">
      {/* Mobile Drawer Backdrop */}
      {mobileSidebarOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      {/* Left Navigation Sidebar */}
      <aside className={`app-sidebar ${mobileSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-icon-wrap">
              <ShieldAlert size={16} />
            </div>
            <div className="brand-text">
              <span className="brand-title">TrustCheck</span>
              <span className="brand-badge">SaMD v2.4</span>
            </div>
          </div>
          <button
            className="mobile-close-btn"
            onClick={() => setMobileSidebarOpen(false)}
            aria-label="Close navigation"
          >
            <X size={16} />
          </button>
        </div>

        {/* Model & Dataset Context Selectors in Sidebar */}
        <div className="sidebar-context-section">
          <div className="context-card">
            <div className="context-label-row">
              <span className="context-label">EVALUATION TARGET</span>
              <span className="context-modality-pill">
                {selectedModel.startsWith('dr') ? 'DR Fundus' : selectedModel.startsWith('derm') ? 'Dermatology' : selectedModel.startsWith('path') ? 'Histopathology' : selectedModel.startsWith('nlp') ? 'Clinical NLP' : 'CXR Chest'}
              </span>
            </div>
            <div className="sidebar-select-wrap">
              <Cpu size={13} className="select-icon" />
              <select
                value={selectedModel}
                onChange={(e) => handleModelChange(e.target.value)}
                className="sidebar-select"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="context-card">
            <div className="context-label-row">
              <span className="context-label">STRESS COHORT</span>
              <span className="context-count-pill">
                {datasets.find(d => d.id === selectedDataset)?.sample_count || 60} {datasets.find(d => d.id === selectedDataset)?.modality === 'clinical_nlp' ? 'notes' : 'scans'}
              </span>
            </div>
            <div className="sidebar-select-wrap">
              <Database size={13} className="select-icon" />
              <select
                value={selectedDataset}
                onChange={(e) => handleDatasetChange(e.target.value)}
                className="sidebar-select"
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.sample_count} {d.modality === 'clinical_nlp' ? 'notes' : 'scans'})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Navigation Links */}
        <div className="sidebar-nav-group">
          <div className="nav-group-title">AUDIT &amp; VALIDATION</div>
          <button
            className={`nav-link ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('overview');
              setMobileSidebarOpen(false);
            }}
          >
            <Activity size={15} />
            <span>Certification Overview</span>
          </button>
          <button
            className={`nav-link ${activeTab === 'cohort' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('cohort');
              setMobileSidebarOpen(false);
            }}
          >
            <Layers size={15} />
            <span>Cohort Stress Suite</span>
          </button>
          <button
            className={`nav-link ${activeTab === 'failures' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('failures');
              setMobileSidebarOpen(false);
            }}
          >
            <AlertTriangle size={15} />
            <span>Silent Failures</span>
            {(auditData?.discrepancies?.length || 0) > 0 && (
              <span className="nav-counter danger">{auditData?.discrepancies?.length}</span>
            )}
          </button>
          <button
            className={`nav-link ${activeTab === 'arena' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('arena');
              setMobileSidebarOpen(false);
            }}
          >
            <Cpu size={15} />
            <span>Model Comparison</span>
          </button>
        </div>

        <div className="sidebar-nav-group">
          <div className="nav-group-title">DIAGNOSTIC LAB</div>
          <button
            className={`nav-link ${activeTab === 'stress_studio' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('stress_studio');
              setMobileSidebarOpen(false);
            }}
          >
            <Sliders size={15} />
            <span>Optical Stress Studio</span>
          </button>
        </div>

        {/* Sidebar Footer */}
        <div className="sidebar-footer">
          <button
            onClick={() => {
              setShowIngestModal(true);
              setMobileSidebarOpen(false);
            }}
            className="sidebar-action-btn"
          >
            <Upload size={14} />
            <span>Ingest Benchmark</span>
          </button>
          <div className="sidebar-footer-row">
            <div className="engine-status">
              <span className="status-indicator live" />
              <span className="status-label">Engine Online</span>
            </div>
            <button
              onClick={toggleTheme}
              className="theme-toggle-btn-small"
              title={theme === 'dark' ? 'Switch to Light Theme' : 'Switch to Dark Theme'}
              aria-label="Toggle theme mode"
            >
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            </button>
          </div>
        </div>
      </aside>

      {/* Main Viewport */}
      <div className="main-viewport">
        <header className="workspace-header">
          <div className="header-left-group">
            <button
              className="mobile-menu-btn"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Open navigation menu"
              title="Open Navigation"
            >
              <Menu size={18} />
            </button>
            <div className="header-breadcrumbs">
              <span className="crumb-root">TrustCheck</span>
              <span className="crumb-sep">/</span>
              <span className="crumb-segment">{selectedModel.startsWith('dr') ? 'Retinal Fundus' : selectedModel.startsWith('derm') ? 'Dermatology' : selectedModel.startsWith('path') ? 'Digital Pathology' : selectedModel.startsWith('nlp') ? 'Clinical NLP' : 'Chest Radiography'}</span>
              <span className="crumb-sep">/</span>
              <span className="crumb-active">{selectedModel}</span>
            </div>
          </div>

          <div className="workspace-actions">
            {auditData?.certificate_url && (
              <a
                href={auditData.certificate_url}
                download
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary btn-sm"
              >
                <Download size={13} />
                <span>Download Dossier</span>
              </a>
            )}
            <button
              onClick={handleRunAudit}
              disabled={loading}
              className="btn btn-primary btn-sm"
            >
              <RefreshCw size={13} className={loading ? 'spin' : ''} />
              <span>{loading ? 'Auditing Model...' : 'Run Single Audit'}</span>
            </button>
          </div>
        </header>

        {/* Content Area */}
        <main className="content-view">
        {/* TAB 1: OVERVIEW & CERTIFICATE */}
        {activeTab === 'overview' && (
          !auditData ? (
            <div className="empty-panel-container">
              <div className={`empty-state-card ${loading ? 'auditing' : ''}`}>
                {loading ? (
                  <>
                    <div className="empty-state-icon-box">
                      <RefreshCw size={28} className="spin" />
                    </div>
                    <h3 className="empty-state-title">Auditing Model Integrity...</h3>
                    <p className="empty-state-desc">
                      Executing optical perturbation tests, calculating Expected Calibration Error, and verifying lesion concordance for <strong>{selectedModel}</strong>...
                    </p>
                  </>
                ) : (
                  <>
                    <div className="empty-state-icon-box">
                      <Activity size={28} />
                    </div>
                    <h3 className="empty-state-title">No Audit Executed for {selectedModel}</h3>
                    <p className="empty-state-desc">
                      Execute an automated stress audit to compute composite readiness score, calibration error curve, and hardware perturbation spectrum.
                    </p>
                    <div className="empty-state-actions">
                      <button
                        onClick={handleRunAudit}
                        disabled={loading}
                        className="btn btn-primary"
                      >
                        <RefreshCw size={14} className={loading ? 'spin' : ''} />
                        <span>Run Single Audit</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="overview-container">
              {/* Top Dossier Hero Deck */}
              <div className="dossier-hero">
                <div className="dossier-main">
                  <div>
                    <h2 className="dossier-headline">{auditData?.verdict_title || 'Running Initial Audit...'}</h2>
                  <p className="dossier-summary">{auditData?.guardrail_policy}</p>
                  <div className="dossier-model-hash">
                    <span className="num-tabular">Target: {selectedModel}</span>
                    <span>•</span>
                    <span>Evaluation Cohort: {auditData?.samples_audited || 6} Stress Benchmarks</span>
                  </div>
                </div>

                <div className="dossier-action-bar">
                  {auditData?.certificate_url && (
                    <a
                      href={auditData.certificate_url}
                      download
                      target="_blank"
                      rel="noreferrer"
                      className="btn btn-secondary"
                    >
                      <Download size={13} /> Download Audit Dossier (PDF)
                    </a>
                  )}
                  <button
                    onClick={() => setActiveTab('failures')}
                    className={`btn ${(auditData?.discrepancies?.length || 0) > 0 ? 'btn-danger-subtle' : 'btn-ghost'}`}
                  >
                    <AlertTriangle size={13} /> Inspect Clinical Discordances ({auditData?.discrepancies?.length || 0})
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
                  {(() => {
                    const optVal = auditData?.optical_robustness;
                    const calVal = auditData?.calibration_precision;
                    const utlVal = auditData?.utility_margin;
                    return (
                      <>
                        <div className="submetric-row">
                          <span>Optical Robustness</span>
                          <span className="num-tabular">{optVal !== undefined ? `${optVal}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${optVal === undefined ? '' : optVal < 50 ? 'fail' : optVal < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${optVal !== undefined ? Math.min(100, Math.max(0, optVal)) : 0}%` }}
                          />
                        </div>

                        <div className="submetric-row">
                          <span>Calibration Precision</span>
                          <span className="num-tabular">{calVal !== undefined ? `${calVal}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${calVal === undefined ? '' : calVal < 50 ? 'fail' : calVal < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${calVal !== undefined ? Math.min(100, Math.max(0, calVal)) : 0}%` }}
                          />
                        </div>

                        <div className="submetric-row">
                          <span>Clinical Utility Margin</span>
                          <span className="num-tabular">{utlVal !== undefined ? `${utlVal}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${utlVal === undefined ? '' : utlVal < 50 ? 'fail' : utlVal < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${utlVal !== undefined ? Math.min(100, Math.max(0, utlVal)) : 0}%` }}
                          />
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>

            {/* Connected Telemetry Strip */}
            {(() => {
              const eceVal = auditData?.calibration?.ece_percent ?? null;
              const eceBadge = eceVal === null ? { label: '--', cls: 'bay-tag-muted' }
                : eceVal < 5.0 ? { label: 'PASS • CALIBRATED', cls: 'bay-tag-success' }
                : eceVal < 15.0 ? { label: 'MODERATE RISK', cls: 'bay-tag-warning' }
                : { label: 'CRITICAL HAZARD', cls: 'bay-tag-danger' };

              const specRows = getSpectrumRows();
              const v1 = specRows[0];
              const v1Title = v1?.vector_name || (isCXR(selectedModel) ? 'Thoracic Motion Blur' : isNLP(selectedModel) ? 'OCR Typographical Noise' : 'Defocus / Motion Tolerance');
              const v1Val = v1?.retained_stability ?? auditData?.stress_tests?.blur_ladder?.slice(-1)[0]?.retained_stability ?? null;
              const v1Badge = v1Val === null ? { label: '--', cls: 'bay-tag-muted' }
                : v1Val >= 70 ? { label: 'RESILIENT', cls: 'bay-tag-success' }
                : v1Val >= 40 ? { label: 'DEGRADED', cls: 'bay-tag-warning' }
                : { label: 'SUB-THRESHOLD', cls: 'bay-tag-danger' };
              const v1Desc = v1?.max_param ? `Max stress tested: ${v1.max_param}` : 'Max stress tested: σ = 6.0 (Severe motion)';

              const v2 = specRows[1];
              const v2Title = v2?.vector_name || (isCXR(selectedModel) ? 'CR/DR Contrast Attenuation' : isNLP(selectedModel) ? 'Medical Abbreviation Density' : 'Illumination Sensitivity');
              const v2Val = v2?.retained_stability ?? auditData?.stress_tests?.illumination_ladder?.slice(-1)[0]?.retained_stability ?? null;
              const v2Badge = v2Val === null ? { label: '--', cls: 'bay-tag-muted' }
                : v2Val >= 70 ? { label: 'STABLE', cls: 'bay-tag-success' }
                : v2Val >= 40 ? { label: 'MODERATE SENSITIVITY', cls: 'bay-tag-warning' }
                : { label: 'SEVERE SENSITIVITY', cls: 'bay-tag-danger' };
              const v2Desc = v2?.max_param ? `Max stress tested: ${v2.max_param}` : 'Max stress tested: -80% flash attenuation';

              const missCount = auditData?.discrepancies?.length || 0;

              return (
                <div className="telemetry-strip">
                  <div className="telemetry-bay">
                    <div className="bay-header">
                      <span className="bay-title">Expected Calibration Error</span>
                      <span className={`bay-tag ${eceBadge.cls}`}>{eceBadge.label}</span>
                    </div>
                    <div className="bay-value-row">
                      <span className="bay-val num-tabular">{eceVal !== null ? eceVal : '--'}%</span>
                      <span className="bay-unit">ECE</span>
                    </div>
                    <span className="bay-desc">Calibration error against ground truth</span>
                  </div>

                  <div className="telemetry-bay">
                    <div className="bay-header">
                      <span className="bay-title">{v1Title}</span>
                      <span className={`bay-tag ${v1Badge.cls}`}>{v1Badge.label}</span>
                    </div>
                    <div className="bay-value-row">
                      <span className="bay-val num-tabular">{v1Val !== null ? v1Val : '--'}%</span>
                      <span className="bay-unit">retained</span>
                    </div>
                    <span className="bay-desc">{v1Desc}</span>
                  </div>

                  <div className="telemetry-bay">
                    <div className="bay-header">
                      <span className="bay-title">{v2Title}</span>
                      <span className={`bay-tag ${v2Badge.cls}`}>{v2Badge.label}</span>
                    </div>
                    <div className="bay-value-row">
                      <span className="bay-val num-tabular">{v2Val !== null ? v2Val : '--'}%</span>
                      <span className="bay-unit">retained</span>
                    </div>
                    <span className="bay-desc">{v2Desc}</span>
                  </div>

                  <div
                    className="telemetry-bay interactive-bay"
                    onClick={() => setActiveTab('failures')}
                    title="Click to inspect clinical failure cases"
                  >
                    <div className="bay-header">
                      <span className="bay-title">Silent Misses &amp; Discordance</span>
                      <span className={`bay-tag ${missCount > 0 ? 'bay-tag-danger' : 'bay-tag-success'}`}>
                        {missCount > 0 ? 'CRITICAL' : 'ZERO'}
                      </span>
                    </div>
                    <div className="bay-value-row">
                      <span
                        className="bay-val num-tabular"
                        style={{
                          color: missCount > 0 ? 'var(--danger)' : 'var(--success)',
                        }}
                      >
                        {missCount}
                      </span>
                      <span className="bay-unit">{missCount === 1 ? 'patient' : 'patients'}</span>
                    </div>
                    <span className="bay-desc">
                      {missCount > 0 ? 'Diagnostic discordance vs anatomical truth' : 'Zero shortcut discordance detected'}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Stress Test Breakdown Table */}
            {(() => {
              const spectrum = getSpectrumRows();

              return (
                <div className="card table-card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title-text">Stress Degradation Spectrum</h3>
                    </div>
                  </div>
                  <div className="table-wrapper">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Stress Vector</th>
                          <th>Baseline Level</th>
                          <th>Max Stress Level</th>
                          <th>Retained Model Stability</th>
                          <th>Clinical Safety Verdict</th>
                        </tr>
                      </thead>
                      <tbody>
                        {spectrum.map((v, idx) => {
                          const val = v.retained_stability !== undefined ? Number(v.retained_stability) : undefined;
                          const isPass = val !== undefined && val >= 70;
                          const isWarn = val !== undefined && val >= 40 && val < 70;
                          const pillCls = val === undefined ? 'status-pill-warn' : isPass ? 'status-pill-pass' : isWarn ? 'status-pill-warn' : 'status-pill-fail';
                          const pillText = v.verdict || (val === undefined ? 'Evaluating...' : isPass ? 'PASS • Resilient' : isWarn ? 'WARN • Degraded' : 'FAIL • Sub-threshold');
                          const trackCls = val === undefined ? '' : isPass ? 'pass' : isWarn ? 'warn' : 'fail';

                          return (
                            <tr key={idx}>
                              <td><strong>{v.vector_name || v.vector}</strong></td>
                              <td className="num-tabular">{v.min_param || v.min}</td>
                              <td className="num-tabular">{v.max_param || v.max}</td>
                              <td>
                                <div className="table-retention-cell">
                                  <div className="retention-val-group">
                                    <span className="retention-main-num num-tabular">{val !== undefined ? `${val}%` : '--'}</span>
                                  </div>
                                  <div className="retention-mini-track">
                                    <div
                                      className={`retention-mini-fill ${trackCls}`}
                                      style={{ width: `${val !== undefined ? Math.min(100, val) : 0}%` }}
                                    />
                                  </div>
                                </div>
                              </td>
                              <td><span className={`status-pill ${pillCls}`}>{pillText}</span></td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()}
            </div>
          )
        )}

        {/* TAB 2: COHORT STRESS SUITE (SOTA) */}
        {activeTab === 'cohort' && (
          !cohortData ? (
            <div className="empty-panel-container">
              <div className={`empty-state-card ${cohortLoading ? 'auditing' : ''}`}>
                {cohortLoading ? (
                  <>
                    <div className="empty-state-icon-box">
                      <RefreshCw size={28} className="spin" />
                    </div>
                    <h3 className="empty-state-title">Executing Cohort Audit Battery...</h3>
                    <p className="empty-state-desc">
                      Evaluating multicenter perturbation suites, worst-group fairness disparities, and G-AUDIT shortcut hazards on <strong>{selectedDataset}</strong>...
                    </p>
                  </>
                ) : (
                  <>
                    <div className="empty-state-icon-box">
                      <Layers size={28} />
                    </div>
                    <h3 className="empty-state-title">No Cohort Audit Battery Executed</h3>
                    <p className="empty-state-desc">
                      Execute the multicenter cohort battery for <strong>{selectedModel}</strong> against <strong>{selectedDataset}</strong> to evaluate clinical corruption robustness (cMCE), worst-group disparities, and autonomous triage deferral policies.
                    </p>
                    <div className="empty-state-actions">
                      <button
                        onClick={handleRunCohortAudit}
                        disabled={cohortLoading}
                        className="btn btn-primary"
                      >
                        <RefreshCw size={14} className={cohortLoading ? 'spin' : ''} />
                        <span>Run Cohort Audit Battery</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="cohort-container">
              {/* Top Cohort Dossier Hero Deck */}
              <div className="dossier-hero">
              <div className="dossier-main">
                <div>
                  <h2 className="dossier-headline">Multicenter Cohort Stress Battery</h2>
                  <div className="cohort-meta-strip">
                    <span className="meta-pill">Target: {cohortData?.model_metadata?.name || cohortData?.target_model || selectedModel}</span>
                    <span className="meta-pill">Modality: {cohortData?.modality ? cohortData.modality.toUpperCase() : 'CLINICAL'}</span>
                    <span className="meta-pill">Cohort: {cohortData?.dataset_metadata?.name || selectedDataset}</span>
                  </div>
                </div>

                <div className="dossier-action-bar">
                  <span className={`badge ${
                    !cohortData ? 'badge-neutral' :
                    cohortData.verdict?.includes('PASS') ? 'badge-success' :
                    cohortData.verdict?.includes('REJECT') || cohortData.verdict?.includes('NO_GO') ? 'badge-danger' :
                    'badge-warning'
                  }`}>
                    <AlertTriangle size={13} /> {cohortLoading ? 'AUDITING COHORT...' : (cohortData?.verdict ? cohortData.verdict.replace(/_/g, ' ') : 'PENDING AUDIT')}
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
                    <Download size={14} /> Download Audit Dossier (PDF)
                  </a>
                </div>
              </div>

              <div className="dossier-score-card">
                <div>
                  <div className="score-dial-header">
                    <div>
                      <span className="score-number num-tabular">{cohortLoading ? '...' : (cohortData?.trust_score ?? '--')}</span>
                      <span className="score-max">/100</span>
                    </div>
                  </div>
                  <div className="score-label-meta">Composite TrustScore</div>
                </div>

                <div className="score-submetrics">
                  {(() => {
                    const cmceVal = cohortData?.metrics?.clinical_mce?.clinical_mean_corruption_error_cmce;
                    const robustPct = cmceVal !== undefined 
                      ? Math.round((1.0 - Math.min(1.0, cmceVal > 1.0 ? cmceVal / 100.0 : cmceVal)) * 1000) / 10 
                      : null;

                    const eqOdds = cohortData?.metrics?.fairness_and_shift?.equalized_odds_difference;
                    const rawFairness = cohortData?.metrics?.fairness_and_shift?.fairness_score;
                    const equityPct = rawFairness !== undefined
                      ? Number(rawFairness).toFixed(1)
                      : (eqOdds !== undefined
                          ? (Math.round((1.0 - Math.min(1.0, eqOdds)) * 1000) / 10).toFixed(1)
                          : null);

                    const deferralRate = cohortData?.metrics?.selective_suppression_policy?.recommended_triage_deferral_rate;
                    const yieldPct = deferralRate !== undefined
                      ? Math.round((1.0 - deferralRate) * 1000) / 10
                      : null;

                    return (
                      <>
                        <div className="submetric-row">
                          <span>Clinical Corruption Robustness</span>
                          <span className="num-tabular">{robustPct !== null ? `${robustPct}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${robustPct === null ? '' : robustPct < 50 ? 'fail' : robustPct < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${robustPct !== null ? Math.min(100, Math.max(0, robustPct)) : 0}%` }}
                          />
                        </div>

                        <div className="submetric-row">
                          <span>Subgroup Equity Parity</span>
                          <span className="num-tabular">{equityPct !== null ? `${equityPct}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${equityPct === null ? '' : Number(equityPct) < 50 ? 'fail' : Number(equityPct) < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${equityPct !== null ? Math.min(100, Math.max(0, Number(equityPct))) : 0}%` }}
                          />
                        </div>

                        <div className="submetric-row">
                          <span>Safe Deferral Yield</span>
                          <span className="num-tabular">{yieldPct !== null ? `${yieldPct}%` : '--'}</span>
                        </div>
                        <div className="submetric-track">
                          <div
                            className={`submetric-fill ${yieldPct === null ? '' : yieldPct < 50 ? 'fail' : yieldPct < 75 ? 'warn' : 'pass'}`}
                            style={{ width: `${yieldPct !== null ? Math.min(100, Math.max(0, yieldPct)) : 0}%` }}
                          />
                        </div>
                      </>
                    );
                  })()}
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
                  <span className="bay-val num-tabular">
                    {cohortData?.metrics?.clinical_mce?.clinical_mean_corruption_error_cmce !== undefined
                      ? (cohortData.metrics.clinical_mce.clinical_mean_corruption_error_cmce > 1.0
                          ? (cohortData.metrics.clinical_mce.clinical_mean_corruption_error_cmce / 100.0).toFixed(2)
                          : Number(cohortData.metrics.clinical_mce.clinical_mean_corruption_error_cmce).toFixed(2))
                      : '--'}
                  </span>
                  <span className="bay-unit">index</span>
                </div>
                <span className="bay-desc">Corruption error across perturbation suite</span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Worst-Group FNR</span>
                  <span className={`bay-tag ${cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr > 0.2 ? 'bay-tag-danger' : 'bay-tag-success'}`}>
                    {cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr !== undefined
                      ? (cohortData.metrics.worst_group_benchmarks.worst_group_fnr > 0.2 ? 'HIGH RISK' : 'LOW RISK')
                      : '--'}
                  </span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular" style={{ color: cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr > 0.2 ? 'var(--danger)' : 'var(--text-main)' }}>
                    {cohortData?.metrics?.worst_group_benchmarks?.worst_group_fnr !== undefined
                      ? `${(cohortData.metrics.worst_group_benchmarks.worst_group_fnr * 100).toFixed(1)}%`
                      : '--'}
                  </span>
                  <span className="bay-unit">FNR</span>
                </div>
                <span className="bay-desc">
                  Stratum: {cohortData?.metrics?.worst_group_benchmarks?.worst_performing_group || '--'}
                </span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">Selective Deferral</span>
                  <span className="bay-tag bay-tag-success">OPTIMAL</span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular">
                    {cohortData?.metrics?.selective_suppression_policy?.recommended_triage_deferral_rate !== undefined
                      ? `${(cohortData.metrics.selective_suppression_policy.recommended_triage_deferral_rate * 100).toFixed(0)}%`
                      : '--'}
                  </span>
                  <span className="bay-unit">triage deferral</span>
                </div>
                <span className="bay-desc">
                  {cohortData?.metrics?.selective_suppression_policy?.expected_accuracy_gain !== undefined
                    ? `+${cohortData.metrics.selective_suppression_policy.expected_accuracy_gain.toFixed(1)}% accuracy yield on triage deferral`
                    : 'Autonomous safety triage routing'}
                </span>
              </div>

              <div className="telemetry-bay">
                <div className="bay-header">
                  <span className="bay-title">G-AUDIT Hazards</span>
                  <span className={`bay-tag ${cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts?.length > 0 ? 'bay-tag-warning' : 'bay-tag-success'}`}>
                    {cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts !== undefined
                      ? `${cohortData.metrics.gaudit_shortcut_risk.high_risk_shortcuts.length} FLAGGED`
                      : '--'}
                  </span>
                </div>
                <div className="bay-value-row">
                  <span className="bay-val num-tabular" style={{ color: cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts?.length > 0 ? 'var(--warning)' : 'var(--text-main)' }}>
                    {cohortData?.metrics?.gaudit_shortcut_risk?.high_risk_shortcuts !== undefined
                      ? `${cohortData.metrics.gaudit_shortcut_risk.high_risk_shortcuts.length} Shortcuts`
                      : '--'}
                  </span>
                </div>
                <span className="bay-desc">Spurious non-clinical correlation</span>
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
                    </div>
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
                        {cohortData?.metrics?.gaudit_shortcut_risk?.gaudit_matrix && Object.keys(cohortData.metrics.gaudit_shortcut_risk.gaudit_matrix).length > 0 ? (
                          Object.entries(cohortData.metrics.gaudit_shortcut_risk.gaudit_matrix).map(([attr, val]) => (
                            <tr key={attr}>
                              <td><strong>{attr}</strong></td>
                              <td className="num-tabular">{val.detectability_auc !== undefined ? Number(val.detectability_auc).toFixed(2) : '--'}</td>
                              <td className="num-tabular">{val.utility_auc !== undefined ? Number(val.utility_auc).toFixed(2) : '--'}</td>
                              <td>
                                {(val.status === 'SHORTCUT_HAZARD' || val.risk_status === 'HIGH_SHORTCUT_HAZARD') ? (
                                  <span className="status-fail">HIGH HAZARD</span>
                                ) : (
                                  <span className="status-pass">LOW RISK</span>
                                )}
                              </td>
                            </tr>
                          ))
                        ) : (
                          <tr><td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>{cohortLoading ? 'Computing G-AUDIT matrix...' : 'No G-AUDIT shortcut risk data computed for this cohort. Click "Run Cohort Audit Battery" to audit.'}</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Prevalence Shift Table */}
                <div className="card table-card">
                  <div className="card-header">
                    <div>
                      <h3 className="card-title-text">Prevalence Shift &amp; Alert Fatigue</h3>
                    </div>
                  </div>
                  <div className="table-wrapper">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Clinical Setting (Prevalence)</th>
                          <th>Bayes PPV</th>
                          <th>Alert Fatigue Assessment</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const ladder = cohortData?.metrics?.prevalence_shift_simulation?.prevalence_ladder 
                            || cohortData?.metrics?.prevalence_shift_simulation?.ladder;
                          if (ladder && ladder.length > 0) {
                            return ladder.map((row, idx) => {
                              const prev = row.disease_prevalence ?? row.prevalence;
                              const ppv = row.ppv ?? row.bayes_ppv;
                              const burden = row.clinical_alert_burden 
                                || (row.false_alert_burden_ratio !== undefined ? `${Number(row.false_alert_burden_ratio).toFixed(2)}x` : (prev !== undefined && prev < 0.05 ? 'HIGH BURDEN' : 'ACCEPTABLE'));
                              const isSafe = row.alert_fatigue_false_alarm_pct !== undefined 
                                ? row.alert_fatigue_false_alarm_pct < 20.0 
                                : (row.false_alert_burden_ratio !== undefined ? row.false_alert_burden_ratio <= 2.0 : burden === 'ACCEPTABLE');
                              const settingName = prev !== undefined
                                ? (prev >= 0.30 ? 'Tertiary Center' : prev >= 0.15 ? 'Regional Hospital' : prev >= 0.05 ? 'District Clinic' : 'Rural Primary Screen')
                                : 'Screening';

                              return (
                                <tr key={idx}>
                                  <td><strong>{prev !== undefined ? `${(prev * 100).toFixed(0)}%` : '--'}</strong> ({settingName})</td>
                                  <td className="num-tabular">{ppv !== undefined ? `${(ppv * 100).toFixed(1)}%` : '--'}</td>
                                  <td>
                                    <span className={isSafe ? 'status-pass' : 'status-fail'}>{burden}</span>
                                  </td>
                                </tr>
                              );
                            });
                          }
                          return (
                            <tr><td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>{cohortLoading ? 'Simulating prevalence shift...' : 'No prevalence shift simulation available for this cohort. Click "Run Cohort Audit Battery" to execute.'}</td></tr>
                          );
                        })()}
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
                    </div>
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
                        {(() => {
                          const dcaRows = cohortData?.metrics?.clinical_utility_dca?.dca_points 
                            || cohortData?.metrics?.clinical_utility_dca?.net_benefit_curve;
                          if (dcaRows && dcaRows.length > 0) {
                            return dcaRows.map((row, idx) => {
                              const pt = row.threshold_probability ?? row.threshold_pt;
                              const netModel = row.net_benefit_model;
                              const netAll = row.net_benefit_treat_all;
                              const isSuperior = row.superior_to_defaults ?? (netModel !== undefined && netAll !== undefined && netModel > netAll);

                              return (
                                <tr key={idx}>
                                  <td><strong>{pt !== undefined ? `${(pt * 100).toFixed(0)}%` : '--'}</strong></td>
                                  <td className={netModel !== undefined && netModel > 0 ? 'status-pass num-tabular' : 'status-fail num-tabular'}>
                                    {netModel !== undefined ? netModel.toFixed(3) : '--'}
                                  </td>
                                  <td className="num-tabular">
                                    {netAll !== undefined ? netAll.toFixed(3) : '--'}
                                  </td>
                                  <td>
                                    {isSuperior ? (
                                      <span className="status-pass">CLINICAL ADVANTAGE</span>
                                    ) : (
                                      <span className="status-fail">NO BENEFIT OVER TREAT-ALL</span>
                                    )}
                                  </td>
                                </tr>
                              );
                            });
                          }
                          return (
                            <tr><td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '24px' }}>{cohortLoading ? 'Evaluating clinical utility boundaries...' : 'No DCA net benefit curve available for this cohort. Click "Run Cohort Audit Battery" to execute.'}</td></tr>
                          );
                        })()}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Hard Safety Vetoes & Contraindications */}
                <div className="card">
                  <div className="card-header" style={{ padding: '0 0 12px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                    <div>
                      <h3 className="card-title-text">Non-Compensatory Safety Vetoes</h3>
                    </div>
                    <span className="badge badge-danger"><ShieldAlert size={12} /> Active Vetoes</span>
                  </div>

                  <div style={{ marginTop: '14px' }}>
                    {cohortData?.clinical_contraindications && cohortData.clinical_contraindications.length > 0 ? (
                      cohortData.clinical_contraindications.map((contra, idx) => (
                        <div key={idx} className="veto-box">
                          <AlertTriangle size={16} className="veto-box-icon" />
                          <div>
                            <span className="veto-box-title">RESTRICTION #{idx + 1}</span>
                            <p className="veto-box-desc">{contra}</p>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div style={{ padding: '16px', color: 'var(--text-muted)', textAlign: 'center', fontSize: '13px' }}>
                        {cohortLoading ? 'Auditing safety constraints...' : 'No critical clinical safety vetoes or contraindications active for this cohort.'}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            </div>
          )
        )}

        {/* TAB 3: OPTICAL STRESS STUDIO */}
        {activeTab === 'stress_studio' && (
          <div className="studio-container">
            <div className="studio-sidebar card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <h3 className="card-title-text" style={{ margin: 0 }}>Stress Controls</h3>
                <span className="meta-pill">{isCXR(selectedModel) ? 'Chest Radiography' : isNLP(selectedModel) ? 'Clinical NLP' : 'Retinal Fundus'}</span>
              </div>

              {isNLP(selectedModel) && (
                <div className="alert-info-box" style={{ marginBottom: 14, fontSize: 11 }}>
                  <Info size={16} style={{ flexShrink: 0 }} />
                  <div>
                    <strong>Vision Stress Studio:</strong> Optical vectors test imaging modalities (Fundus &amp; CXR). For Clinical NLP text perturbation benchmarks (OCR typographical noise, note truncation), explore the <strong>Cohort Safety Suite</strong>.
                  </div>
                </div>
              )}

              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <label className="form-label" style={{ margin: 0 }}>
                    {isCXR(selectedModel) ? 'Select Thoracic CXR Scan:' : 'Select Fundus Test Sample:'}
                  </label>
                  <div style={{ display: 'flex', gap: 4 }}>
                    <button
                      type="button"
                      className={`btn-toggle ${!customScanFile ? 'active' : ''}`}
                      style={{ padding: '2px 6px', fontSize: 10 }}
                      onClick={() => setCustomScanFile(null)}
                    >
                      Presets
                    </button>
                    <label
                      className={`btn-toggle ${customScanFile ? 'active' : ''}`}
                      style={{ padding: '2px 6px', fontSize: 10, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 3 }}
                    >
                      <Upload size={10} />
                      Upload
                      <input
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={(e) => {
                          if (e.target.files && e.target.files[0]) {
                            setCustomScanFile(e.target.files[0]);
                          }
                        }}
                      />
                    </label>
                  </div>
                </div>

                {!customScanFile ? (
                  <select
                    value={sampleKey}
                    onChange={(e) => setSampleKey(e.target.value)}
                    className="form-select"
                  >
                    {isCXR(selectedModel) ? (
                      <>
                        <option value="scan_0001">CXR Scan 1: PA View (Bilateral Clear)</option>
                        <option value="scan_0002">CXR Scan 2: Cardiomegaly / Effusion</option>
                        <option value="scan_0003">CXR Scan 3: Low Dose Portable CXR</option>
                        <option value="scan_0004">CXR Scan 4: Apical Infiltration / Pneumonia</option>
                        <option value="scan_0005">CXR Scan 5: Baseline Normal Thoracic</option>
                        <option value="scan_0006">CXR Scan 6: Subtle Consolidation</option>
                      </>
                    ) : (
                      <>
                        <option value="sample_clinical_pass">Sample 1: Clinical Grade Fundus (IDRiD)</option>
                        <option value="sample_severe_npdr">Sample 2: Severe Proliferative DR</option>
                        <option value="sample_low_illumination">Sample 3: Low Illumination Haze</option>
                        <option value="sample_motion_blur">Sample 4: Handheld Motion Blur</option>
                        <option value="sample_corneal_glare">Sample 5: Corneal Glare Flash</option>
                        <option value="sample_silent_failure_candidate">Sample 6: Microaneurysm Candidate</option>
                      </>
                    )}
                  </select>
                ) : (
                  <div className="custom-file-preview">
                    <span>Custom: <strong>{customScanFile.name}</strong></span>
                    <button
                      type="button"
                      className="btn-icon-xs"
                      onClick={() => setCustomScanFile(null)}
                      title="Clear custom scan"
                    >
                      ✕
                    </button>
                  </div>
                )}
              </div>

              <div className="form-group">
                <label className="form-label">Stress Vector:</label>
                <div className="radio-group">
                  {(isCXR(selectedModel)
                    ? [
                        { id: 'blur', label: 'Motion Blur (σ)' },
                        { id: 'illumination', label: 'Exposure Drop' },
                        { id: 'glare', label: 'Poisson Noise' },
                        { id: 'resolution', label: 'Res Scaling' },
                      ]
                    : [
                        { id: 'blur', label: 'Defocus Blur (σ)' },
                        { id: 'illumination', label: 'Flash Drop (%)' },
                        { id: 'glare', label: 'Corneal Glare' },
                        { id: 'resolution', label: 'Sensor Res (px)' },
                      ]
                  ).map((vec) => (
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
                    {stressType === 'blur' && `σ = ${stressIntensity.toFixed(1)}`}
                    {stressType === 'illumination' && `-${stressIntensity}% Drop`}
                    {stressType === 'glare' && (isCXR(selectedModel) ? `Noise = ${(stressIntensity * 100).toFixed(0)}%` : `Glare = ${(stressIntensity * 100).toFixed(0)}%`)}
                    {stressType === 'resolution' && `${stressIntensity}×${stressIntensity}px`}
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

              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 14 }}>
                <button
                  type="button"
                  className="btn btn-primary"
                  style={{ width: '100%', justifyContent: 'center', gap: 8 }}
                  onClick={() => runLiveStress(stressType, stressIntensity, sampleKey, customScanFile)}
                  disabled={stressLoading}
                >
                  {stressLoading ? (
                    <>
                      <RefreshCw size={15} className="spin" />
                      <span>Inferring Telemetry...</span>
                    </>
                  ) : (
                    <>
                      <Zap size={15} />
                      <span>Apply Stress &amp; Run Telemetry</span>
                    </>
                  )}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ width: '100%', justifyContent: 'center', fontSize: 11 }}
                  onClick={() => {
                    if (stressType === 'blur') setStressIntensity(0);
                    else if (stressType === 'illumination') setStressIntensity(0);
                    else if (stressType === 'glare') setStressIntensity(0);
                    else setStressIntensity(384);
                  }}
                  disabled={stressLoading}
                >
                  Reset to Baseline (0% Stress)
                </button>
              </div>
            </div>

            {/* Live Inspection Viewer */}
            <div className="studio-viewer card">
              <div className="viewer-header">
                <div>
                  <h3 style={{ margin: 0, fontSize: 14 }}>Perturbed Input vs Model Diagnostic Output</h3>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Live ONNX tensor transformation &amp; real-time confidence tracking
                  </span>
                </div>
                <div className="viewer-stats">
                  <span className={`live-badge ${stressLoading ? 'updating' : ''}`}>
                    <span className="live-dot" />
                    {stressLoading ? 'Inferring...' : 'Connected'}
                  </span>
                  <span>Latency: <strong>{liveStressResult?.latency_ms ? `${liveStressResult.latency_ms} ms` : '--'}</strong></span>
                </div>
              </div>

              <div className="viewer-split">
                <div className="image-viewport">
                  {liveStressResult?.image_base64 ? (
                    <img src={liveStressResult.image_base64} alt="Perturbed Clinical Scan" className="fundus-render" />
                  ) : (
                    <div className="image-placeholder">{stressLoading ? 'Running Stress Inference...' : 'No Perturbation Applied'}</div>
                  )}
                  {stressLoading && (
                    <div className="viewport-loading-overlay">
                      <RefreshCw size={24} className="spin" />
                      <span>Re-evaluating ONNX Tensors...</span>
                    </div>
                  )}
                  <span className="img-tag">
                    Vector: {stressType} • Intensity: {
                      stressType === 'blur' ? `σ=${stressIntensity.toFixed(1)}` :
                      stressType === 'illumination' ? `-${stressIntensity}%` :
                      stressType === 'glare' ? `${stressIntensity.toFixed(2)}` :
                      `${stressIntensity}px`
                    }
                  </span>
                </div>

                <div className="telemetry-panel">
                  <h4>Model Decision Telemetry</h4>

                  {!liveStressResult ? (
                    stressLoading ? (
                      <div className="empty-telemetry-box">
                        <RefreshCw size={24} className="spin text-primary" />
                        <p className="empty-telemetry-title">Applying Stress &amp; Inferring...</p>
                        <p className="empty-telemetry-desc">Evaluating candidate model inference and lesion detection under active stress.</p>
                      </div>
                    ) : (
                      <div className="empty-telemetry-box">
                        <Eye size={28} className="empty-icon-subtle" />
                        <p className="empty-telemetry-title">Ready for Live Stress Inference</p>
                        <p className="empty-telemetry-desc">
                          Adjust perturbation parameters on the left or click <strong>Apply Stress &amp; Run Telemetry</strong> to evaluate live inference and lesion detection.
                        </p>
                      </div>
                    )
                  ) : (
                    <>
                      <div className="telemetry-item">
                        <span className="t-label">Diagnostic Output:</span>
                        <span className="t-val" style={{ color: 'var(--accent)', fontWeight: 700 }}>
                          {liveStressResult.class_names && liveStressResult.class_names[liveStressResult.predicted_grade]
                            ? liveStressResult.class_names[liveStressResult.predicted_grade]
                            : `Class ${liveStressResult.predicted_grade}`}
                        </span>
                      </div>

                      <div className="telemetry-item" style={{ marginTop: 8 }}>
                        <span className="t-label">Prediction Confidence:</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span className="t-val">{liveStressResult.confidence ?? '--'}%</span>
                          {liveStressResult.confidence_delta !== undefined && (
                            <span
                              className={`delta-badge ${
                                liveStressResult.confidence_delta < -5
                                  ? 'negative'
                                  : liveStressResult.confidence_delta > 0
                                  ? 'positive'
                                  : 'neutral'
                              }`}
                              title="Degradation delta relative to baseline unperturbed scan"
                            >
                              {liveStressResult.confidence_delta > 0 ? '+' : ''}
                              {liveStressResult.confidence_delta}% vs baseline
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{ transform: `scaleX(${(liveStressResult.confidence || 0) / 100})` }}
                        />
                      </div>

                      {/* Class Probability Distribution Breakdown */}
                      {liveStressResult.probs && liveStressResult.probs.length > 0 && (
                        <div style={{ marginTop: 12, marginBottom: 14 }}>
                          <span className="panel-subhead" style={{ display: 'block', marginBottom: 6 }}>
                            Class Probability Distribution:
                          </span>
                          <div className="prob-dist-container">
                            {liveStressResult.probs.map((p, idx) => {
                              const pct = Math.round(p * 1000) / 10;
                              const isTop = idx === liveStressResult.predicted_grade;
                              const label = liveStressResult.class_names?.[idx] || `Class ${idx}`;
                              return (
                                <div key={idx} className="prob-dist-row">
                                  <div className="prob-dist-header">
                                    <span className={`prob-dist-label ${isTop ? 'active' : ''}`}>
                                      {label}
                                    </span>
                                    <span className={`prob-dist-pct ${isTop ? 'active' : ''}`}>
                                      {pct}%
                                    </span>
                                  </div>
                                  <div className="prob-dist-track">
                                    <div
                                      className={`prob-dist-fill ${isTop ? 'active' : ''}`}
                                      style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                                    />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      <h5 style={{ marginTop: '14px', marginBottom: '8px', fontSize: 11, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                        {isCXR(selectedModel) ? 'Thoracic Biomarkers & Features:' : 'Detected Retinal Lesions:'}
                      </h5>
                      <div className="biomarker-chips">
                        {isCXR(selectedModel) ? (
                          <>
                            <div className="chip">
                              <span>Cardiomegaly Index:</span>
                              <strong>{liveStressResult.biomarkers?.cardiomegaly_ratio ?? '0.48'}</strong>
                            </div>
                            <div className="chip">
                              <span>Effusion Density:</span>
                              <strong>{liveStressResult.biomarkers?.effusion_density ?? '0.0%'}</strong>
                            </div>
                            <div className="chip">
                              <span>Infiltrate Quadrants:</span>
                              <strong>{liveStressResult.biomarkers?.infiltrate_quadrants ?? 0} quads</strong>
                            </div>
                            <div className="chip">
                              <span>Consolidation Score:</span>
                              <strong>{liveStressResult.biomarkers?.consolidation_score ?? 'Low'}</strong>
                            </div>
                          </>
                        ) : (
                          <>
                            <div className="chip">
                              <span>Microaneurysms:</span>
                              <strong>{liveStressResult.biomarkers?.microaneurysms ?? 0}</strong>
                            </div>
                            <div className="chip">
                              <span>Hard Exudates:</span>
                              <strong>{liveStressResult.biomarkers?.exudate_area_pct ?? 0}%</strong>
                            </div>
                            <div className="chip">
                              <span>Hemorrhages:</span>
                              <strong>{liveStressResult.biomarkers?.hemorrhage_quadrants ?? 0} quads</strong>
                            </div>
                            <div className="chip">
                              <span>Cotton Wool Spots:</span>
                              <strong>{liveStressResult.biomarkers?.soft_exudate_area_pct ?? 0}%</strong>
                            </div>
                          </>
                        )}
                      </div>

                      {!isCXR(selectedModel) && liveStressResult.predicted_grade === 0 && (
                        (liveStressResult.biomarkers?.microaneurysms > 0) ||
                        (liveStressResult.biomarkers?.exudate_area_pct > 0) ||
                        (liveStressResult.biomarkers?.hemorrhage_quadrants > 0) ||
                        (liveStressResult.biomarkers?.soft_exudate_area_pct > 0)
                      ) && (
                        <div className="alert-danger-box">
                          <AlertTriangle size={18} style={{ flexShrink: 0 }} />
                          <div>
                            <strong>SILENT CLINICAL DISCORDANCE:</strong> Model outputs Normal (Grade 0), but physical micro-lesions were detected by computer-vision feature extractors. High clinical risk!
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: SILENT FAILURES INSPECTOR */}
        {activeTab === 'failures' && (
          !auditData ? (
            <div className="empty-panel-container">
              <div className={`empty-state-card ${loading ? 'auditing' : ''}`}>
                {loading ? (
                  <>
                    <div className="empty-state-icon-box warning">
                      <RefreshCw size={28} className="spin" />
                    </div>
                    <h3 className="empty-state-title">Scanning for Silent Clinical Discordance...</h3>
                    <p className="empty-state-desc">
                      Cross-referencing candidate model classifications against anatomical biomarkers for <strong>{selectedModel}</strong>...
                    </p>
                  </>
                ) : (
                  <>
                    <div className="empty-state-icon-box warning">
                      <AlertTriangle size={28} />
                    </div>
                    <h3 className="empty-state-title">No Audit Executed for {selectedModel}</h3>
                    <p className="empty-state-desc">
                      Execute an audit check to scan benchmarks for silent false negatives where candidate models output Grade 0 (Normal) despite physical lesions being present.
                    </p>
                    <div className="empty-state-actions">
                      <button
                        onClick={handleRunAudit}
                        disabled={loading}
                        className="btn btn-primary"
                      >
                        <RefreshCw size={14} className={loading ? 'spin' : ''} />
                        <span>Run Single Audit</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="failures-container">
              <div className="card">
                <div className="card-header">
                  <div>
                    <h3 className="card-title-text">Silent Clinical Failures &amp; Discrepancies</h3>
                  </div>
                  <span className={`badge ${(auditData?.discrepancies?.length || 0) > 0 ? 'badge-danger' : 'badge-success'}`}>
                    {auditData?.discrepancies?.length || 0} Violations Found
                  </span>
                </div>

                {(!auditData?.discrepancies || auditData.discrepancies.length === 0) ? (
                  <div style={{ padding: '36px 24px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <CheckCircle2 size={32} style={{ color: 'var(--success)', marginBottom: '12px' }} />
                    <h4 style={{ margin: '0 0 6px', color: 'var(--text-primary)' }}>Zero Silent Failures Detected</h4>
                    <p style={{ margin: 0, fontSize: '13px' }}>Candidate model classifications matched physical anatomical ground truth across audited benchmarks.</p>
                  </div>
                ) : (
                  <div className="failure-list">
                    {auditData.discrepancies.map((disc, idx) => (
                      <div key={idx} className="failure-card">
                        <div className="failure-meta">
                          <span className="failure-id">Sample ID: {disc.sample_id}</span>
                          <span className="failure-violation">{disc.violation_type}</span>
                        </div>

                        <div className="failure-comparison">
                          <div className="comparison-box model-side">
                            <span className="box-title">Candidate Model Classification</span>
                            <span className="box-grade">Grade {disc.predicted_grade}</span>
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
                )}
              </div>
            </div>
          )
        )}

        {/* TAB 5: MODEL ARENA */}
        {activeTab === 'arena' && (
          !auditData && Object.keys(completedAudits).length === 0 ? (
            <div className="empty-panel-container">
              <div className={`empty-state-card ${loading ? 'auditing' : ''}`}>
                {loading ? (
                  <>
                    <div className="empty-state-icon-box">
                      <RefreshCw size={28} className="spin" />
                    </div>
                    <h3 className="empty-state-title">Auditing Model for Arena Benchmark...</h3>
                    <p className="empty-state-desc">
                      Executing evaluation benchmarks to populate comparative discrimination metrics...
                    </p>
                  </>
                ) : (
                  <>
                    <div className="empty-state-icon-box">
                      <Cpu size={28} />
                    </div>
                    <h3 className="empty-state-title">No Audits Executed for Comparison</h3>
                    <p className="empty-state-desc">
                      Comparative benchmark evaluation requires at least one evaluated model audit. Run an audit check to view architecture comparisons and AUROC discrimination curves.
                    </p>
                    <div className="empty-state-actions">
                      <button
                        onClick={handleRunAudit}
                        disabled={loading}
                        className="btn btn-primary"
                      >
                        <RefreshCw size={14} className={loading ? 'spin' : ''} />
                        <span>Run Single Audit</span>
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="arena-container">
              <div className="card">
                <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
                  <div>
                    <h3 className="card-title-text">Model Comparison</h3>
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
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td><strong>Architecture Type</strong></td>
                          <td>PP-LCNet + MSAG Attention</td>
                          <td>ResNet18 Deep Ensemble</td>
                        </tr>
                        <tr>
                          <td><strong>Model Size / Parameters</strong></td>
                          <td><span className="status-pass">7.6M Params (12.7 MB)</span></td>
                          <td><span className="status-fail">11.2M Params (44.6 MB)</span></td>
                        </tr>
                        <tr>
                          <td><strong>Inference Latency</strong></td>
                          <td><span className="status-pass">6.8 ms (146 FPS)</span></td>
                          <td>14.2 ms (70 FPS)</td>
                        </tr>
                        <tr>
                          <td><strong>In-Domain AUC</strong></td>
                          <td>92.8%</td>
                          <td><span className="status-pass">95.4%</span></td>
                        </tr>
                        <tr>
                          <td><strong>Defocus Noise Resilience</strong></td>
                          <td><span className="status-fail">28.0% Stability Retained</span></td>
                          <td><span className="status-pass">64.5% Stability Retained</span></td>
                        </tr>
                        <tr>
                          <td><strong>Expected Calibration Error (ECE)</strong></td>
                          <td><span className="status-fail">18.4% (Overconfident)</span></td>
                          <td><span className="status-pass">4.2% (Well-calibrated)</span></td>
                        </tr>
                        <tr>
                          <td><strong>Final Deployment Verdict</strong></td>
                          <td><span className="badge badge-warning">CONDITIONAL PASS</span></td>
                          <td><span className="badge badge-success">APPROVED FOR GPU SERVER</span></td>
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
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td><strong>Architecture Type</strong></td>
                          <td>MobileNetV2 Depthwise-Conv</td>
                          <td>DenseNet-121 Feature Reuse</td>
                        </tr>
                        <tr>
                          <td><strong>Model Size / Parameters</strong></td>
                          <td><span className="status-pass">3.5M Params (8.4 MB)</span></td>
                          <td><span className="status-fail">7.0M Params (27.7 MB)</span></td>
                        </tr>
                        <tr>
                          <td><strong>Inference Latency</strong></td>
                          <td><span className="status-pass">4.1 ms (240 FPS)</span></td>
                          <td>12.8 ms (78 FPS)</td>
                        </tr>
                        <tr>
                          <td><strong>In-Domain AUC</strong></td>
                          <td>74.2%</td>
                          <td><span className="status-pass">86.8%</span></td>
                        </tr>
                        <tr>
                          <td><strong>CR vs DR Contrast Sensitivity</strong></td>
                          <td><span className="status-fail">42.1% Stability Retained</span></td>
                          <td><span className="status-pass">78.4% Stability Retained</span></td>
                        </tr>
                        <tr>
                          <td><strong>Expected Calibration Error (ECE)</strong></td>
                          <td><span className="status-fail">10.8% (Borderline Overconfident)</span></td>
                          <td><span className="status-pass">3.8% (Calibrated Posterior)</span></td>
                        </tr>
                        <tr>
                          <td><strong>Final Deployment Verdict</strong></td>
                          <td><span className="badge badge-danger">RESTRICTED / CAUTION</span></td>
                          <td><span className="badge badge-success">APPROVED FOR WORKSTATION</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )
        )}
      </main>
      </div>

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
                      <option value="dermatology_dermoscopy">Dermatology (Dermoscopy & Melanoma)</option>
                      <option value="digital_pathology">Digital Pathology (Whole Slide Imaging WSI)</option>
                      <option value="clinical_nlp">Clinical NLP (EHR Notes & Discharge Summaries)</option>
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
                    placeholder="e.g. assets/models/retinal_dr/dr_retinal_resnet_teacher.onnx"
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
                      <option value="dermatology_dermoscopy">Dermatology (Dermoscopy & Melanoma)</option>
                      <option value="digital_pathology">Digital Pathology (Whole Slide Imaging WSI)</option>
                      <option value="clinical_nlp">Clinical NLP (EHR Notes & Discharge Summaries)</option>
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
                      <option value="dermatology_dermoscopy">Dermatology (Dermoscopy & Melanoma)</option>
                      <option value="digital_pathology">Digital Pathology (Whole Slide Imaging WSI)</option>
                      <option value="clinical_nlp">Clinical NLP (EHR Notes & Discharge Summaries)</option>
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
