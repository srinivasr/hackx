import React, { useState, useEffect, useRef } from 'react';
import {
  Database,
  Cpu,
  FileText,
  ShieldCheck,
  Activity,
  ArrowRight,
  ArrowLeft,
  RotateCcw
} from 'lucide-react';

const archSteps = [
  {
    stepNum: '01',
    shortTitle: 'Cohort & Target',
    title: 'Clinical Cohort & Target Intake',
    phaseBadge: 'Phase 01: Ingestion & Sandboxing',
    roleTag: 'Model Registry • Zero-Egress Sandbox',
    statusTag: 'ISOLATED RUNTIME',
    icon: <Database size={40} style={{ color: '#38bdf8' }} />,
    summary: 'The evaluation lifecycle initiates by pairing a candidate clinical model (e.g. CheXNet, Swin-Radiology) with a verified multicenter dataset cohort (MIMIC-CXR, CheXpert). The model weights and inference runtime are deployed inside an air-gapped sandbox without external network egress to prevent Protected Health Information (PHI) leakage.',
    mechanisms: [
      {
        title: 'Demographic Stratification',
        desc: 'The baseline cohort is stratified across biological sex, age deciles (<40, 40-65, >65), scanner types, and clinical acuity levels (ICU vs. Outpatient triage).'
      },
      {
        title: 'Clean Baseline Benchmarking',
        desc: 'TrustCheck executes forward inference across uncorrupted scans to establish clean diagnostic sensitivity (TPR >= 95.0%) and baseline Expected Calibration Error (ECE <= 0.05).'
      },
      {
        title: 'Format & Weight Normalization',
        desc: 'Supports PyTorch, ONNX, and TorchScript formats, mapping target output logits into standardized clinical diagnostic categories.'
      }
    ],
    artifactLabel: 'SANDBOX INTAKE SCHEMA (JSON)',
    artifactCode: `{
  "candidate_model": "CheXNet_DenseNet121_v2",
  "modality": "Chest Radiography (DICOM/PNG)",
  "cohort_source": "MIMIC-CXR-Multicenter",
  "stratified_attributes": ["biological_sex", "age_deciles", "patient_acuity"],
  "baseline_empirical_metrics": {
    "baseline_sensitivity": 0.964,
    "baseline_calibration_ece": 0.048,
    "clean_accuracy": 0.942
  }
}`
  },
  {
    stepNum: '02',
    shortTitle: 'AWS Strands',
    title: 'Autonomous Adversarial Red-Teaming',
    phaseBadge: 'Phase 02: Dynamic Stress Interrogation',
    roleTag: 'AWS Strands Agent • Local LLM Orchestrator',
    statusTag: 'AGENTIC LOOP ACTIVE',
    icon: <Cpu size={40} style={{ color: '#a855f7' }} />,
    summary: 'AWS Strands operates as our autonomous Red-Teaming interrogator. Unlike traditional static testing scripts, Strands is driven by a local LLM agentic reasoning loop (Qwen 2.5 7B via Ollama). It autonomously inspects candidate behavior, selects adversarial attack vectors, and iteratively stress-tests clinical edge cases.',
    mechanisms: [
      {
        title: 'Fast Gradient Sign Method (FGSM / PGD)',
        desc: 'Computes pixel loss gradients to inject imperceptible adversarial noise (epsilon = 0.05), testing resilience against hardware sensor artifacts, motion blur, and contrast drift.'
      },
      {
        title: 'Subgroup Fairness Audit (Equalized Odds)',
        desc: 'Probes diagnostic sensitivity across protected demographic groups to reveal hidden demographic disparities (e.g. False Negative Rate disparities across biological sexes).'
      },
      {
        title: 'Biomarker Consensus Multi-Model Probing',
        desc: 'Cross-verifies predictions against clinical ensemble models to flag Silent False Negatives—high-confidence classifications that completely miss critical lesions.'
      }
    ],
    artifactLabel: 'STRANDS AUTONOMOUS REASONING & EXECUTION TRACE',
    artifactCode: `[Strands Agent Reasoning]: "Candidate model displays strong global accuracy, but exhibits potential vulnerability under subtle sensor noise. Initiating FGSM perturbation..."
[Strands Tool Execution] => run_adversarial_fgsm(model="CheXNet", epsilon=0.05)
[Strands Tool Execution] => run_subgroup_fairness_audit(strata=["Sex", "Age"])
[Strands Tool Execution] => verify_biomarker_consensus(tolerance=0.05)
[Strands Telemetry] => Sensitivity drop under FGSM: 14.8% | Subgroup FNR gap: 7.1%`
  },
  {
    stepNum: '03',
    shortTitle: 'Metrics Payload',
    title: 'Standardized Telemetry Synthesis',
    phaseBadge: 'Phase 03: Telemetry Serialization',
    roleTag: 'Audit Payload • Basis Point Precision',
    statusTag: 'TAMPER-EVIDENT PAYLOAD',
    icon: <FileText size={40} style={{ color: '#f59e0b' }} />,
    summary: 'Following adversarial interrogation, Strands aggregates all empirical metrics into a machine-readable, tamper-evident JSON payload. Crucially, all floating-point rates and percentages are converted into basis points (bp = rate * 100) to enforce strict 64-bit integer determinism in policy evaluation.',
    mechanisms: [
      {
        title: 'Basis Point Scaling (*_bp)',
        desc: 'Converts 96.4% sensitivity to 9640 bp. Eliminates floating-point rounding errors and cross-platform non-determinism during policy execution.'
      },
      {
        title: 'Shortcut Vulnerability Index (SVI)',
        desc: 'Measures model reliance on non-causal confounding artifacts, such as radiologist orientation markers, chest tube artifacts, or scanner department stamps.'
      },
      {
        title: 'Zero-Tolerance Clinical Flags',
        desc: 'Flags underpowered subgroups and records the exact count of silent false negatives as strict integer assertions.'
      }
    ],
    artifactLabel: 'SERIALIZED AUDIT PAYLOAD PASSED TO CEDAR',
    artifactCode: `{
  "sensitivity_bp": 9640,
  "shortcut_vulnerability_index_bp": 820,
  "subgroup_fnr_disparity_bp": 710,
  "expected_calibration_error_bp": 540,
  "silent_false_negatives": 0,
  "adversarial_resilience_drop_bp": 1480,
  "underpowered_subgroup": false
}`
  },
  {
    stepNum: '04',
    shortTitle: 'AWS Cedar (The Judge)',
    title: 'Deterministic Policy Evaluation',
    phaseBadge: 'Phase 04: Policy-as-Code Governance',
    roleTag: 'AWS Cedar Engine • Provable Safety Rules',
    statusTag: 'ZERO-HALLUCINATION JUDGE',
    icon: <ShieldCheck size={40} style={{ color: '#10b981' }} />,
    summary: 'The crux of TrustCheck governance is AWS Cedar, an enterprise-grade Policy-as-Code engine. Unlike subjective LLM evaluators that hallucinate, Cedar is mathematically verifiable, executes in under 5 milliseconds, and enforces strict separation of clinical policy from ML engineering logic.',
    mechanisms: [
      {
        title: 'Typed Entity Architecture',
        desc: 'Structures hospital governance into formal entities: Principal (Hospital::Role::"ChiefMedicalOfficer"), Action (Action::"CertifyDeployment"), and Resource (Hospital::Department::"EmergencyICU").'
      },
      {
        title: 'Department-Specific Guardrail Thresholds',
        desc: 'High-acuity units like the ICU enforce higher sensitivity (>= 9500 bp) and lower calibration tolerance (<= 800 bp) than general triage units.'
      },
      {
        title: 'Default-Deny & Schema Enforcement',
        desc: 'Contains explicit forbid rules that automatically reject models if any telemetry attribute is missing or if demographic subgroups are underpowered.'
      }
    ],
    artifactLabel: 'HOSPITAL PACS CEDAR POLICY (cedar/policies/hospital_pacs.cedar)',
    artifactCode: `permit(
  principal,
  action == Action::"CertifyDeployment",
  resource
)
when {
  context.sensitivity_bp >= resource.required_sensitivity_bp &&
  context.shortcut_vulnerability_index_bp <= resource.max_shortcut_vulnerability_bp &&
  context.subgroup_fnr_disparity_bp <= resource.max_fnr_disparity_bp &&
  context.expected_calibration_error_bp <= resource.max_ece_bp &&
  context.silent_false_negatives == 0
};

// Explicit default-deny if critical telemetry is missing
forbid(principal, action == Action::"CertifyDeployment", resource)
when {
  !(context has sensitivity_bp) ||
  !(context has subgroup_fnr_disparity_bp) ||
  !(context has silent_false_negatives) ||
  context.adversarial_resilience_drop_bp > resource.max_adversarial_drop_bp
};`
  },
  {
    stepNum: '05',
    shortTitle: 'Deployment Verdict',
    title: 'Autonomous Deployment Verdict & Enforcement',
    phaseBadge: 'Phase 05: Gate Verdict & Cryptographic Dossier',
    roleTag: 'Zero-Trust Gate • Cryptographic Audit Trail',
    statusTag: 'GOVERNANCE ENFORCED',
    icon: <Activity size={40} style={{ color: '#6366f1' }} />,
    summary: 'In the final stage, AWS Cedar outputs a definitive PERMIT or FORBID authorization. This decision is strictly enforced at the inference gateway, physically gating whether the model is routed into hospital PACS or locked out of clinical workflows.',
    mechanisms: [
      {
        title: 'Physical PACS Gate Enforcement',
        desc: 'On FORBID, the inference gateway immediately drops the deployment request, preventing unsafe or biased models from diagnosing live patients.'
      },
      {
        title: 'Automated Diagnostic Attribution',
        desc: 'Upon denial, Cedar provides exact policy reason identifiers (e.g. policy_subgroup_fairness), pinpointing the failure mode for engineering teams.'
      },
      {
        title: 'Cryptographic Compliance Dossier',
        desc: 'On pass, TrustCheck generates an immutable, SHA-256 signed audit certificate documenting all empirical metrics and Cedar proofs for FDA/EU-AI Act compliance.'
      }
    ],
    artifactLabel: 'AUTONOMOUS GATE AUTHORIZATION RESULT (JSON)',
    artifactCode: `{
  "decision": "ALLOW",
  "gate_status": "CERTIFIED_FOR_CLINICAL_DEPLOYMENT",
  "diagnostics": {
    "reasons": ["policy_0"],
    "errors": []
  },
  "department": "Hospital::Department::EmergencyICU",
  "dossier_hash": "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069",
  "pacs_routing": "ENABLED"
}`
  }
];

export default function AboutArchitectureDeck({ activeStep = 0, onStepChange }) {
  const [currentStep, setCurrentStep] = useState(activeStep);
  const deckRef = useRef(null);
  const lastWheelTime = useRef(0);
  const touchStartY = useRef(null);

  useEffect(() => {
    setCurrentStep(activeStep);
  }, [activeStep]);

  const goToStep = (idx) => {
    const nextIdx = Math.max(0, Math.min(idx, archSteps.length - 1));
    setCurrentStep(nextIdx);
    onStepChange?.(nextIdx);
  };

  // Intercept scroll wheel on deck container to switch pages in-place
  useEffect(() => {
    const el = deckRef.current;
    if (!el) return;

    const handleWheel = (e) => {
      if (Math.abs(e.deltaY) < 18) return;

      // Check if user is scrolling inside an internal scrollable element (e.g. card-text-side)
      const scrollableChild = e.target.closest('.card-text-side');
      if (scrollableChild) {
        const canScrollDown = scrollableChild.scrollTop + scrollableChild.clientHeight < scrollableChild.scrollHeight - 6;
        const canScrollUp = scrollableChild.scrollTop > 6;
        if (e.deltaY > 0 && canScrollDown) return;
        if (e.deltaY < 0 && canScrollUp) return;
      }

      const now = Date.now();
      // Cooldown of 400ms to prevent jumping multiple pages on one swipe
      if (now - lastWheelTime.current < 400) {
        e.preventDefault();
        return;
      }

      if (e.deltaY > 0) {
        // Scroll downward -> switch to next page
        if (currentStep < archSteps.length - 1) {
          e.preventDefault();
          lastWheelTime.current = now;
          goToStep(currentStep + 1);
        }
      } else if (e.deltaY < 0) {
        // Scroll upward -> switch to previous page
        if (currentStep > 0) {
          e.preventDefault();
          lastWheelTime.current = now;
          goToStep(currentStep - 1);
        }
      }
    };

    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [currentStep]);

  // Touch navigation
  useEffect(() => {
    const el = deckRef.current;
    if (!el) return;

    const onTouchStart = (e) => {
      touchStartY.current = e.touches[0].clientY;
    };

    const onTouchEnd = (e) => {
      if (touchStartY.current === null) return;
      const deltaY = touchStartY.current - e.changedTouches[0].clientY;
      touchStartY.current = null;
      if (Math.abs(deltaY) > 40) {
        if (deltaY > 0 && currentStep < archSteps.length - 1) {
          goToStep(currentStep + 1);
        } else if (deltaY < 0 && currentStep > 0) {
          goToStep(currentStep - 1);
        }
      }
    };

    el.addEventListener('touchstart', onTouchStart, { passive: true });
    el.addEventListener('touchend', onTouchEnd, { passive: true });
    return () => {
      el.removeEventListener('touchstart', onTouchStart);
      el.removeEventListener('touchend', onTouchEnd);
    };
  }, [currentStep]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown') {
        goToStep(currentStep + 1);
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
        goToStep(currentStep - 1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentStep]);

  const step = archSteps[currentStep];

  return (
    <div className="architecture-deck-viewport" ref={deckRef}>
      {/* Header Description */}
      <div style={{ textAlign: 'center', marginBottom: '14px', marginTop: '4px' }}>
        <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', marginBottom: '8px' }}>
          <img
            src="/logo.png"
            alt="TrustCheck"
            style={{
              width: '46px',
              height: '46px',
              objectFit: 'contain',
              filter: 'drop-shadow(0 4px 14px rgba(16, 185, 129, 0.25))'
            }}
          />
        </div>
        <h2 className="dossier-headline" style={{ marginBottom: '6px' }}>
          TrustCheck Architecture
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '13.5px', maxWidth: '640px', margin: '0 auto' }}>
          A 5-phase zero-trust pipeline pairing autonomous LLM red-teaming (AWS Strands) with deterministic policy-as-code authorization (AWS Cedar).
        </p>
      </div>

      {/* Horizontal Pipeline Stepper */}
      <div className="stepper-pipeline">
        {archSteps.map((s, idx) => (
          <React.Fragment key={idx}>
            <div
              className={`arch-node ${currentStep === idx ? 'active-node' : ''}`}
              onClick={() => goToStep(idx)}
              title={`Switch to ${s.title}`}
            >
              <div className="arch-node-badge">0{idx + 1}</div>
              {React.cloneElement(s.icon, { size: 18 })}
              <span>{s.shortTitle}</span>
            </div>
            {idx < archSteps.length - 1 && (
              <div className={`arch-edge ${currentStep > idx ? 'completed-edge' : ''}`}>
                →
              </div>
            )}
          </React.Fragment>
        ))}
      </div>

      {/* In-Place Active Stage Card */}
      <div key={currentStep} className="architecture-stage-card">
        <div className="card-grid-layout">
          {/* Left Side: Visual Identity */}
          <div className="card-visual-side">
            <span className="card-step-num">STAGE {step.stepNum}</span>
            <div className="card-visual-icon-wrap">
              {step.icon}
            </div>
            <h3>{step.title}</h3>
            <div className="card-role-tag">{step.roleTag}</div>
            <div className="card-status-pill">
              <span className="dot" />
              <span>{step.statusTag}</span>
            </div>
          </div>

          {/* Right Side: Deep-Dive Content */}
          <div className="card-text-side">
            <div className="card-phase-header">
              <span className="card-phase-badge">{step.phaseBadge}</span>
              <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Stage {currentStep + 1} of {archSteps.length}
              </span>
            </div>

            <p>{step.summary}</p>

            {/* Core Mechanisms */}
            <div className="card-mechanisms">
              <div className="card-mechanisms-title">Core Architectural Mechanisms</div>
              {step.mechanisms.map((m, mIdx) => (
                <div key={mIdx} className="card-mechanism-item">
                  <span className="card-mechanism-bullet">•</span>
                  <div>
                    <strong style={{ color: 'var(--text-primary)' }}>{m.title}: </strong>
                    <span>{m.desc}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Code / Payload Artifact */}
            <div className="card-artifact-box">
              <div className="card-artifact-header">
                <span>{step.artifactLabel}</span>
                <span>{currentStep === 3 ? 'CEDAR' : currentStep === 1 ? 'LOG' : 'JSON'}</span>
              </div>
              <pre><code>{step.artifactCode}</code></pre>
            </div>

            {/* Navigation Footer */}
            <div className="card-nav-footer">
              <div className="card-progress-dots">
                {archSteps.map((_, dotIdx) => (
                  <div
                    key={dotIdx}
                    className={`card-progress-dot ${dotIdx === currentStep ? 'active' : ''}`}
                    onClick={() => goToStep(dotIdx)}
                    style={{ cursor: 'pointer' }}
                    title={`Switch to stage ${dotIdx + 1}`}
                  />
                ))}
              </div>

              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Scroll down or use arrow keys to switch
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                {currentStep > 0 && (
                  <button
                    className="card-advance-btn"
                    onClick={() => goToStep(currentStep - 1)}
                  >
                    <ArrowLeft size={13} />
                    <span>Previous</span>
                  </button>
                )}

                {currentStep < archSteps.length - 1 ? (
                  <button
                    className="card-advance-btn"
                    onClick={() => goToStep(currentStep + 1)}
                  >
                    <span>Next: {archSteps[currentStep + 1].shortTitle}</span>
                    <ArrowRight size={13} />
                  </button>
                ) : (
                  <button
                    className="card-advance-btn"
                    onClick={() => goToStep(0)}
                  >
                    <RotateCcw size={13} />
                    <span>Restart Tour</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
