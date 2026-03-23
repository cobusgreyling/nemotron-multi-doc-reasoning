# Nemotron Multi-Document Reasoning

Comparative document analysis with transparent chain-of-thought using NVIDIA Nemotron Super 49B.

![Demo Screenshot](images/demo1.png)

## What It Does

Upload or paste two documents — term sheets, vendor proposals, employment offers, insurance policies — and the model produces a structured comparative analysis with full reasoning trace visible.

Four analysis modes:
- **Comparative Risk Assessment** — Risk matrix across financial, operational, legal, and flexibility dimensions
- **Side-by-Side Clause Comparison** — Every clause mapped and classified
- **Due Diligence Red Flags** — Ambiguities, one-sided terms, missing protections
- **Financial Impact Analysis** — Quantified costs and worst-case scenarios

## Quick Start

```bash
export NVIDIA_API_KEY="your-key-here"
pip install -r requirements.txt
python app.py
```

Open http://localhost:7865

## Pre-loaded Examples

| Example | What It Tests |
|---------|--------------|
| Series A vs Series B Term Sheets | Liquidation preferences, anti-dilution, board composition, protective provisions |
| Cloud Migration Proposals | Fixed-price vs T&M, SLA comparison, team composition, warranty terms |
| Employment Offer vs Counter-offer | Cash vs equity, severance, non-competes, acceleration clauses |
| Standard vs Premium Cyber Insurance | Coverage limits, exclusions, sub-limits, ransomware coverage |

## Model

Uses NVIDIA Nemotron Super 49B via NIM API with chain-of-thought reasoning enabled. Reasoning trace is streamed separately and displayed in real-time.

## Blog

See [blog.md](blog.md) for the full write-up.
