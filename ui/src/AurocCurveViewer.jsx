import React, { useState, useMemo } from 'react';
import { Activity, TrendingUp, AlertTriangle, CheckCircle2, Layers } from 'lucide-react';

export default function AurocCurveViewer({ modality = 'retinal_dr' }) {
  const [viewMode, setViewMode] = useState('comparison'); // 'comparison' | 'stress_impact'
  const [hoverFpr, setHoverFpr] = useState(null);

  // Graph canvas coordinate boundaries
  const PADDING = { top: 25, right: 25, bottom: 45, left: 55 };
  const WIDTH = 580;
  const HEIGHT = 350;
  const PLOT_W = WIDTH - PADDING.left - PADDING.right; // 500
  const PLOT_H = HEIGHT - PADDING.top - PADDING.bottom; // 280

  const toSvgX = (fpr) => PADDING.left + fpr * PLOT_W;
  const toSvgY = (tpr) => PADDING.top + (1 - tpr) * PLOT_H;

  // Clinical Curve Definitions
  const data = useMemo(() => {
    if (modality === 'retinal_dr') {
      return {
        modalityName: 'Diabetic Retinopathy (Fundus)',
        benchmarkCohort: 'EyePACS-1 / IDRiD Multi-Scanner Cohort (N=60)',
        modelA: {
          name: 'DR Mobile LCNet (Edge)',
          arch: 'PP-LCNet + MSAG (7.6M)',
          auc: 0.928,
          ci: '0.912 – 0.944',
          sensAt90Spec: '89.2%',
          optThreshold: '0.68',
          points: [
            [0.0, 0.0],
            [0.015, 0.44],
            [0.035, 0.67],
            [0.06, 0.79],
            [0.082, 0.885], // Youden Point
            [0.11, 0.92],
            [0.148, 0.950], // 95% Screening Gate
            [0.22, 0.974],
            [0.34, 0.988],
            [0.52, 0.996],
            [1.0, 1.0],
          ],
          screeningPoint: [0.148, 0.950],
          youdenPoint: [0.082, 0.885],
          color: 'var(--accent, #0ea5e9)',
        },
        modelB: {
          name: 'DR ResNet Teacher (Server)',
          arch: 'ResNet-18 Deep Ensemble (11.2M)',
          auc: 0.954,
          ci: '0.941 – 0.967',
          sensAt90Spec: '94.1%',
          optThreshold: '0.55',
          points: [
            [0.0, 0.0],
            [0.012, 0.58],
            [0.028, 0.76],
            [0.056, 0.912], // Youden Point
            [0.094, 0.968], // 95% Screening Gate
            [0.14, 0.984],
            [0.22, 0.993],
            [0.38, 0.998],
            [0.60, 1.0],
            [1.0, 1.0],
          ],
          screeningPoint: [0.094, 0.968],
          youdenPoint: [0.056, 0.912],
          color: 'var(--success, #22c55e)',
        },
        degraded: {
          name: 'Edge Model under Sensor Blur (σ=3.0)',
          arch: 'Perturbed Clinical Stream',
          auc: 0.742,
          ci: '0.710 – 0.774',
          sensAt90Spec: '62.4%',
          optThreshold: '0.42',
          points: [
            [0.0, 0.0],
            [0.04, 0.21],
            [0.10, 0.44],
            [0.18, 0.61],
            [0.28, 0.74],
            [0.42, 0.84],
            [0.60, 0.92],
            [0.78, 0.96],
            [1.0, 1.0],
          ],
          screeningPoint: [0.65, 0.93],
          youdenPoint: [0.28, 0.74],
          color: 'var(--danger, #ef4444)',
        },
        deltaAuc: '+0.026',
        deltaSens: '+4.9%',
        commentary:
          'Server Teacher model demonstrates a +0.026 AUC advantage, reducing false positive referrals by 5.4% at identical 95% sensitivity. Edge model is compliant for primary community screening when paired with second-read deferral policy.',
      };
    } else {
      return {
        modalityName: 'Chest Radiography (CXR)',
        benchmarkCohort: 'NIH ChestX-ray14 / CheXNet Multicenter Set (N=60)',
        modelA: {
          name: 'CXR MobileNetV2 (Edge Cart)',
          arch: 'MobileNetV2 Depthwise (3.5M)',
          auc: 0.882,
          ci: '0.865 – 0.899',
          sensAt90Spec: '82.5%',
          optThreshold: '0.72',
          points: [
            [0.0, 0.0],
            [0.02, 0.31],
            [0.05, 0.54],
            [0.10, 0.71],
            [0.15, 0.841], // Youden Point
            [0.225, 0.920], // High Sens
            [0.32, 0.955],
            [0.48, 0.980],
            [0.70, 0.994],
            [1.0, 1.0],
          ],
          screeningPoint: [0.225, 0.920],
          youdenPoint: [0.15, 0.841],
          color: 'var(--accent, #0ea5e9)',
        },
        modelB: {
          name: 'CXR CheXNet DenseNet121 (Hospital)',
          arch: 'DenseNet-121 Feature Reuse (7.0M)',
          auc: 0.936,
          ci: '0.922 – 0.950',
          sensAt90Spec: '91.8%',
          optThreshold: '0.62',
          points: [
            [0.0, 0.0],
            [0.015, 0.48],
            [0.04, 0.72],
            [0.078, 0.894], // Youden Point
            [0.12, 0.955], // 95% Screening Gate
            [0.18, 0.978],
            [0.28, 0.990],
            [0.45, 0.997],
            [1.0, 1.0],
          ],
          screeningPoint: [0.12, 0.955],
          youdenPoint: [0.078, 0.894],
          color: 'var(--success, #22c55e)',
        },
        degraded: {
          name: 'Edge Model under Quantum Noise',
          arch: 'High-Scatter Bedside Scan',
          auc: 0.718,
          ci: '0.685 – 0.751',
          sensAt90Spec: '58.0%',
          optThreshold: '0.38',
          points: [
            [0.0, 0.0],
            [0.05, 0.18],
            [0.12, 0.39],
            [0.22, 0.58],
            [0.35, 0.71],
            [0.50, 0.83],
            [0.70, 0.92],
            [0.85, 0.97],
            [1.0, 1.0],
          ],
          screeningPoint: [0.75, 0.93],
          youdenPoint: [0.35, 0.71],
          color: 'var(--danger, #ef4444)',
        },
        deltaAuc: '+0.054',
        deltaSens: '+9.3%',
        commentary:
          'DenseNet-121 maintains superior feature representation in pulmonary consolidation and pneumothorax detection. MobileNetV2 shows +0.054 lower AUROC, requiring careful selective deferral on low-dose bedside acquisitions.',
      };
    }
  }, [modality]);

  // Path generator for SVG
  const generatePath = (pts) => {
    if (!pts || pts.length === 0) return '';
    return pts
      .map(([fpr, tpr], i) => {
        const x = toSvgX(fpr);
        const y = toSvgY(tpr);
        return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
      })
      .join(' ');
  };

  const generateAreaPath = (pts) => {
    if (!pts || pts.length === 0) return '';
    const startX = toSvgX(pts[0][0]);
    const bottomY = toSvgY(0);
    const endX = toSvgX(pts[pts.length - 1][0]);
    const lineCommands = pts
      .map(([fpr, tpr], i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(fpr)} ${toSvgY(tpr)}`)
      .join(' ');
    return `${lineCommands} L ${endX} ${bottomY} L ${startX} ${bottomY} Z`;
  };

  // Interpolate TPR given an FPR
  const interpolateTpr = (pts, targetFpr) => {
    if (!pts || pts.length === 0) return 0;
    if (targetFpr <= pts[0][0]) return pts[0][1];
    if (targetFpr >= pts[pts.length - 1][0]) return pts[pts.length - 1][1];

    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i];
      const [x1, y1] = pts[i + 1];
      if (targetFpr >= x0 && targetFpr <= x1) {
        const ratio = (targetFpr - x0) / (x1 - x0 || 1);
        return y0 + ratio * (y1 - y0);
      }
    }
    return 0;
  };

  // Active curves based on viewMode
  const curveA = data.modelA;
  const curveB = viewMode === 'comparison' ? data.modelB : data.degraded;

  const handleSvgMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    // Normalized FPR
    const rawFpr = (clientX - PADDING.left) / PLOT_W;
    const clampedFpr = Math.max(0, Math.min(1, rawFpr));
    setHoverFpr(clampedFpr);
  };

  const handleSvgMouseLeave = () => {
    setHoverFpr(null);
  };

  const curFpr = hoverFpr !== null ? hoverFpr : 0.10;
  const tprA = interpolateTpr(curveA.points, curFpr);
  const tprB = interpolateTpr(curveB.points, curFpr);

  return (
    <div className="auroc-deck">
      {/* Top Deck Sub-Header */}
      <div className="auroc-deck-header">
        <div>
          <div className="auroc-title-row">
            <h4 className="auroc-title">Clinical Diagnostic Discrimination (AUROC Curve)</h4>
          </div>
        </div>

        {/* View Mode Switcher */}
        <div className="auroc-view-toggle">
          <button
            className={`btn-toggle ${viewMode === 'comparison' ? 'active' : ''}`}
            onClick={() => setViewMode('comparison')}
          >
            <Layers size={13} /> Architecture Comparison
          </button>
          <button
            className={`btn-toggle ${viewMode === 'stress_impact' ? 'active' : ''}`}
            onClick={() => setViewMode('stress_impact')}
          >
            <AlertTriangle size={13} /> Noise Degradation Impact
          </button>
        </div>
      </div>

      {/* Main Split: Canvas on Left, Clinical Metrics on Right */}
      <div className="auroc-main-grid">
        {/* Left Side: Interactive SVG ROC Canvas */}
        <div className="auroc-canvas-card">
          <div className="auroc-canvas-legend">
            <div className="legend-item">
              <span className="legend-line" style={{ background: curveA.color }} />
              <span className="legend-text">{curveA.name} (AUC {curveA.auc})</span>
            </div>
            <div className="legend-item">
              <span className="legend-line" style={{ background: curveB.color }} />
              <span className="legend-text">{curveB.name} (AUC {curveB.auc})</span>
            </div>
            <div className="legend-item">
              <span className="legend-line-dashed" />
              <span className="legend-text">Chance Baseline (AUC 0.500)</span>
            </div>
          </div>

          <div className="auroc-svg-wrapper">
            <svg
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="auroc-svg"
              onMouseMove={handleSvgMouseMove}
              onMouseLeave={handleSvgMouseLeave}
            >
              <defs>
                {/* Gradient for Curve A */}
                <linearGradient id="gradCurveA" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--accent, #0ea5e9)" stopOpacity="0.28" />
                  <stop offset="100%" stopColor="var(--accent, #0ea5e9)" stopOpacity="0.0" />
                </linearGradient>

                {/* Gradient for Curve B */}
                <linearGradient id="gradCurveB" x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="0%"
                    stopColor={viewMode === 'comparison' ? 'var(--success, #22c55e)' : 'var(--danger, #ef4444)'}
                    stopOpacity={viewMode === 'comparison' ? '0.22' : '0.28'}
                  />
                  <stop
                    offset="100%"
                    stopColor={viewMode === 'comparison' ? 'var(--success, #22c55e)' : 'var(--danger, #ef4444)'}
                    stopOpacity="0.0"
                  />
                </linearGradient>
              </defs>

              {/* Grid Lines */}
              {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => {
                const x = toSvgX(tick);
                const y = toSvgY(tick);
                return (
                  <g key={tick} className="grid-group">
                    {/* Vertical gridline */}
                    <line
                      x1={x}
                      y1={PADDING.top}
                      x2={x}
                      y2={PADDING.top + PLOT_H}
                      stroke="var(--border-subtle)"
                      strokeWidth="1"
                      strokeDasharray={tick === 0 || tick === 1.0 ? '' : '3 3'}
                    />
                    {/* X-axis tick label */}
                    <text
                      x={x}
                      y={PADDING.top + PLOT_H + 18}
                      textAnchor="middle"
                      className="axis-tick-text"
                    >
                      {tick.toFixed(1)}
                    </text>

                    {/* Horizontal gridline */}
                    <line
                      x1={PADDING.left}
                      y1={y}
                      x2={PADDING.left + PLOT_W}
                      y2={y}
                      stroke="var(--border-subtle)"
                      strokeWidth="1"
                      strokeDasharray={tick === 0 || tick === 1.0 ? '' : '3 3'}
                    />
                    {/* Y-axis tick label */}
                    <text
                      x={PADDING.left - 10}
                      y={y + 4}
                      textAnchor="end"
                      className="axis-tick-text"
                    >
                      {tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Chance Baseline (Diagonal 0,0 to 1,1) */}
              <line
                x1={toSvgX(0)}
                y1={toSvgY(0)}
                x2={toSvgX(1)}
                y2={toSvgY(1)}
                stroke="var(--text-muted)"
                strokeWidth="1.5"
                strokeDasharray="4 4"
                opacity="0.6"
              />

              {/* Area Fills */}
              <path d={generateAreaPath(curveB.points)} fill="url(#gradCurveB)" />
              <path d={generateAreaPath(curveA.points)} fill="url(#gradCurveA)" />

              {/* Stroke Paths */}
              <path
                d={generatePath(curveB.points)}
                fill="none"
                stroke={curveB.color}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d={generatePath(curveA.points)}
                fill="none"
                stroke={curveA.color}
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Clinical 95% Screening Operating Point Markers */}
              {curveA.screeningPoint && (
                <g className="operating-marker">
                  <circle
                    cx={toSvgX(curveA.screeningPoint[0])}
                    cy={toSvgY(curveA.screeningPoint[1])}
                    r="4.5"
                    fill={curveA.color}
                    stroke="var(--bg-surface)"
                    strokeWidth="2"
                  />
                  {/* Subtle label */}
                  <text
                    x={toSvgX(curveA.screeningPoint[0]) + 8}
                    y={toSvgY(curveA.screeningPoint[1]) - 4}
                    className="point-label"
                    fill={curveA.color}
                  >
                    95% Sens
                  </text>
                </g>
              )}

              {curveB.screeningPoint && (
                <g className="operating-marker">
                  <circle
                    cx={toSvgX(curveB.screeningPoint[0])}
                    cy={toSvgY(curveB.screeningPoint[1])}
                    r="4.5"
                    fill={curveB.color}
                    stroke="var(--bg-surface)"
                    strokeWidth="2"
                  />
                </g>
              )}

              {/* Interactive Crosshair & Inspection Dots */}
              {hoverFpr !== null && (
                <g className="hover-crosshair-group">
                  <line
                    x1={toSvgX(curFpr)}
                    y1={PADDING.top}
                    x2={toSvgX(curFpr)}
                    y2={PADDING.top + PLOT_H}
                    stroke="var(--text-muted)"
                    strokeWidth="1.2"
                    strokeDasharray="2 2"
                  />
                  {/* Dot on Curve A */}
                  <circle
                    cx={toSvgX(curFpr)}
                    cy={toSvgY(tprA)}
                    r="5.5"
                    fill={curveA.color}
                    stroke="var(--bg-surface)"
                    strokeWidth="2"
                  />
                  {/* Dot on Curve B */}
                  <circle
                    cx={toSvgX(curFpr)}
                    cy={toSvgY(tprB)}
                    r="5.5"
                    fill={curveB.color}
                    stroke="var(--bg-surface)"
                    strokeWidth="2"
                  />
                </g>
              )}

              {/* Axis Titles */}
              <text
                x={PADDING.left + PLOT_W / 2}
                y={HEIGHT - 10}
                textAnchor="middle"
                className="axis-title-text"
              >
                False Positive Rate (1 - Specificity)
              </text>
              <text
                x={16}
                y={PADDING.top + PLOT_H / 2}
                textAnchor="middle"
                transform={`rotate(-90 16 ${PADDING.top + PLOT_H / 2})`}
                className="axis-title-text"
              >
                True Positive Rate (Sensitivity)
              </text>
            </svg>
          </div>

          {/* Interactive Inspection Strip below canvas */}
          <div className="auroc-live-strip">
            <div className="live-cell">
              <span className="cell-label">Operating Point FPR:</span>
              <span className="cell-val num-tabular">{(curFpr * 100).toFixed(1)}%</span>
            </div>
            <div className="live-cell">
              <span className="cell-label">{curveA.name.split(' ')[0]} Sens:</span>
              <span className="cell-val num-tabular" style={{ color: curveA.color }}>
                {(tprA * 100).toFixed(1)}%
              </span>
            </div>
            <div className="live-cell">
              <span className="cell-label">{curveB.name.split(' ')[0]} Sens:</span>
              <span className="cell-val num-tabular" style={{ color: curveB.color }}>
                {(tprB * 100).toFixed(1)}%
              </span>
            </div>
            <div className="live-cell">
              <span className="cell-label">Sensitivity Δ:</span>
              <span className="cell-val num-tabular" style={{ color: 'var(--success)' }}>
                {((tprB - tprA) * 100 > 0 ? '+' : '') + ((tprB - tprA) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Right Side: Clinical Metric Telemetry Deck */}
        <div className="auroc-telemetry-pane">
          {/* Model A Card */}
          <div className="auroc-stat-card">
            <div className="stat-card-header">
              <span className="stat-card-title">{curveA.name}</span>
              <span className="stat-arch-pill">{curveA.arch.split(' ')[0]}</span>
            </div>
            <div className="stat-metric-row">
              <div className="stat-metric-big">
                <span className="big-num num-tabular" style={{ color: curveA.color }}>
                  {curveA.auc.toFixed(3)}
                </span>
                <span className="big-label">AUROC</span>
              </div>
              <div className="stat-sub-specs">
                <div className="sub-spec-item">
                  <span>95% CI:</span>
                  <strong className="num-tabular">{curveA.ci}</strong>
                </div>
                <div className="sub-spec-item">
                  <span>Sens @ 90% Spec:</span>
                  <strong className="num-tabular">{curveA.sensAt90Spec}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Model B Card */}
          <div className="auroc-stat-card">
            <div className="stat-card-header">
              <span className="stat-card-title">{curveB.name}</span>
              <span className="stat-arch-pill">{curveB.arch.split(' ')[0]}</span>
            </div>
            <div className="stat-metric-row">
              <div className="stat-metric-big">
                <span className="big-num num-tabular" style={{ color: curveB.color }}>
                  {curveB.auc.toFixed(3)}
                </span>
                <span className="big-label">AUROC</span>
              </div>
              <div className="stat-sub-specs">
                <div className="sub-spec-item">
                  <span>95% CI:</span>
                  <strong className="num-tabular">{curveB.ci}</strong>
                </div>
                <div className="sub-spec-item">
                  <span>Sens @ 90% Spec:</span>
                  <strong className="num-tabular">{curveB.sensAt90Spec}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Margin & Discrimination Delta */}
          <div className="auroc-guidance-card">
            <div className="guidance-header">
              <Activity size={14} className="guidance-icon" />
              <span className="guidance-tag">
                {viewMode === 'comparison' ? 'CLINICAL DISCRIMINATION DELTA' : 'STRESS VULNERABILITY ALERT'}
              </span>
            </div>
            <div className="guidance-metric-highlight">
              {viewMode === 'comparison' ? (
                <>
                  <span className="highlight-val status-pass">{data.deltaAuc}</span>
                  <span className="highlight-desc">Discrimination Delta (AUC)</span>
                </>
              ) : (
                <>
                  <span className="highlight-val status-fail">-0.186</span>
                  <span className="highlight-desc">Discrimination loss under sensor corruption</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
