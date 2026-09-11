# DEMO.md: TrustCheck 3-Minute Presentation Script

Problem Statement: HLT-08 (Making Healthcare AI Safe to Deploy)
Target Audience: Hospital Chief Medical Officers, AI Safety Auditors, Clinical Committees
Live Command: `./start.sh 3` (FastAPI backend + Vite Web UI at `http://localhost:5173`)

## Phase 1 (0:00 to 1:00): The Paper Accuracy Trap and Deployment Readiness

Presenter Script:
"Over 90% of healthcare deep learning models with high benchmark accuracy in research papers fail catastrophically in real hospitals. Academic benchmarks test on clean, curated data. Real clinics have dirty lenses, corneal flash glare, sensor shifts, and demographic bias. TrustCheck is an independent crash-testing harness for healthcare AI."

Actions:
1. Open `http://localhost:5173` on the Deployment Certification tab.
2. Select Candidate-Model-A (Mobile / Edge LCNet).
3. Review the Hospital Readiness Score.
4. Note the conditional pass verdict badge and explain the hard veto mechanism:
"Even if a model scores high on average, if underdiagnosis disparity exceeds 2.0x or silent false negatives exceed 3%, the model is hard-locked from unconditional clinical deployment."

## Phase 2 (1:00 to 2:15): SOTA Research Cohort Intelligence

Presenter Script:
"We translate ten foundational clinical AI safety papers directly into live audit engines. Let us inspect the 60-patient multicenter cohort."

Actions:
1. Click the Cohort Safety Suite (SOTA) tab.
2. Review the four clinical panels with the evaluation committee:
- G-AUDIT Shortcut Matrix (FDA CDRH / JHU 2025): Show the attribute detectability vs utility table. Explain how TrustCheck detects if a model diagnoses pathology based on scanner model or patient demographics rather than tissue pathology.
- Decision Curve Analysis (Vickers & Elkin 2006): Show the Net Benefit curve across clinical decision thresholds (10% to 40%), proving whether the model provides more clinical utility than treat-all or treat-none policies.
- Prevalence Shift Simulator (Wong et al., JAMA 2021): Show how positive predictive value collapses as disease prevalence drops from secondary hospitals (20%) to rural screening camps (2%), triggering alert fatigue.
- Selective Prediction Suppression (JAMIA 2023): Show how calculating an optimal uncertainty threshold silences borderline predictions and routes them to human clinician dual-read.

## Phase 3 (2:15 to 3:00): Real-Time Stress Studio and Regulatory Dossier

Presenter Script:
"Now let us watch what happens when a model faces low-cost rural hardware live."

Actions:
1. Click the Optical Stress Studio tab.
2. Select Sample Severe NPDR and set perturbation to Corneal Glare.
3. Drag the intensity slider from 0.1 to 0.85:
- Point to the live preview and watch model confidence drop under glare.
- Switch to Silent Failures Inspector to show how the lesion consensus engine catches an active microvascular hemorrhage that the classifier head missed.
4. Click Download FDA SaMD Safety Certificate:
- Open the generated PDF to show the publication-grade vector certificate with regulatory compliance metrics, radar charts, auditor signatures, and Predetermined Change Control Plan (PCCP) triggers.

Closing Line:
"TrustCheck bridges the gap between academic accuracy and clinical deployment safety. It ensures no AI touches a patient until it passes the crash test."
