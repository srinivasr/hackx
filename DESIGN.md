# DESIGN.md — TrustCheck Design System

## Visual Direction: Clinical Precision Instrument

The aesthetic is inspired by certified biomedical equipment, surgical monitors, and high-assurance laboratory telemetry. It avoids both the consumer-tech AI slop (purple glows, floating cards) and decorative fluff. Every pixel communicates safety, diagnostic threshold, or risk.

## Color Palette Tokens
- `--bg-canvas`: `#0A0E17` (Deep Obsidian Void)
- `--bg-surface`: `#111827` (Precision Equipment Gray)
- `--bg-surface-elevated`: `#1A2333` (Active / Hover Surface)
- `--border-subtle`: `#1F2B3F` (Subtle Dividing Rule)
- `--border-focus`: `#3B82F6` (Operational Focus Ring)
- `--text-primary`: `#F8FAFC` (High-contrast Clinical Reading, $\ge 12:1$)
- `--text-secondary`: `#94A3B8` (Secondary Telemetry, $\ge 5:1$)
- `--text-muted`: `#64748B` (Metadata & Axis Labels)
- `--status-danger`: `#DC2626` / `--status-danger-bg`: `rgba(220, 38, 38, 0.12)`
- `--status-warning`: `#D97706` / `--status-warning-bg`: `rgba(217, 119, 6, 0.12)`
- `--status-success`: `#059669` / `--status-success-bg`: `rgba(5, 150, 105, 0.12)`
- `--accent-cyan`: `#0EA5E9` (Active Laser / Scan Reticle Accent)

## Typography Hierarchy
- **Primary Interface**: Inter (400 regular, 500 medium, 600 semibold, 700 bold).
- **Clinical Data & Metrics**: JetBrains Mono with `font-variant-numeric: tabular-nums lining-nums`.
- **Rhythm**:
  - Display Verdict: `24px / 1.2` (bold, tracking `-0.02em`)
  - Section Headings: `14px / 1.4` (semibold, uppercase tracking `0.04em`)
  - Standard Interface: `13px / 1.5`
  - Secondary Metadata: `11px / 1.4`

## Craft Rules (Impeccable Compliance)
1. **No Eyebrows / Kickers**: Headings carry their own weight; no redundant label tags above headers.
2. **Tabular Figures**: All percentages, milliseconds, and confidence levels use mono tabular numbers to prevent jitter on live slider dragging.
3. **Themed Browser Surfaces**:
   - `::selection`: background `#1E3A8A`, color `#FFFFFF`.
   - Custom sleek scrollbar (`8px` track, `#1E293B` thumb).
   - Clear `:focus-visible` outline for keyboard navigation.
4. **Interactive Telemetry**: Fast dynamic slider responses without layout reflow or image jumping.
