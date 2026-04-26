# Web Redesign — Design Spec
**Date:** 2026-04-26
**Branch:** `web-redesign`
**Scope:** Visual polish of `docs/` (index.html, style.css, script.js). No structural changes, no new data sources, no backend. GitHub Pages compatible.

---

## Constraints

- Semi-static: GitHub Pages, no server, no env vars, no exposed keys.
- Preserve existing aesthetic: background tile image, dark blue colour identity, buy/sell green/red signals, news ticker, predictions grid.
- Existing data sources unchanged: `predictions.json` + `rss2json` proxy for news.

---

## Section 1 — Typography

**Fonts (Google Fonts, loaded in `<head>`):**
- `DM Serif Display` (400) — h1, h2 headings
- `Inter` (400, 500, 600) — all body text, labels, UI elements

**Rules:**
- `body` font-family: `'Inter', Arial, sans-serif`
- `h1`, `h2` font-family: `'DM Serif Display', serif`
- h1: 32px, tightened letter-spacing (-0.01em)
- h2: 18px — weight contrast comes from DM Serif vs Inter body, not bold
- Probability line in cards: Inter, 12px, monospace feel via `font-variant-numeric: tabular-nums`
- `Courier New` stays for `.math` spans (existing, correct)
- Footer: Inter, 13px

---

## Section 2 — Prediction Cards

**Structure (unchanged HTML, restyled):**
```
[card]
  [pred-top]   company name — DM Serif Display 20px, darkblue
  [pred-bottom] probability line — Inter 12px, muted
```

**Styling changes:**
- **Top border accent:** `border-top: 3px solid <buy/sell colour>` instead of full-colour bottom strip
- **Bottom strip background:** very light tint — `#f0fdf4` (buy) / `#fff5f5` (sell) instead of solid green/red
- **Card background:** white; no change to border-radius or shadow
- **Name:** DM Serif Display, 20px, darkblue — primary read
- **Probability:** Inter 12px, `color: #555`, tabular nums — secondary read
- **Hover:** existing lift + shadow stays, shadow colour warmed slightly (`rgba(0,0,0,0.12)`)

---

## Section 3 — Layout & Spacing

- **Max-width:** `1200px` on a centred wrapper around all page content
- **Section gaps:** standardise to `24px` margins between sections (currently `30px` inconsistent)
- **Preds container:** `#f7f7f7` background (warmer than current `#f0f0f0`)
- **News ticker:** zero top-margin gap from preds — visually attached as a continuation
- **Info columns:** `align-items: stretch` so both cols match height; cards inside use `height: 100%`
- **Footer:** `margin: 0 0 28px 0`, Inter 13px

---

## Out of Scope (Option B — noted for later)

- Glass-morphism cards
- Probability progress bar
- Heavy card restructuring

---

## Files Changed

| File | Changes |
|------|---------|
| `docs/index.html` | Add Google Fonts `<link>`, add max-width wrapper div |
| `docs/style.css` | Font vars, card border accent, tint backgrounds, spacing, max-width |
| `docs/script.js` | No changes needed |
