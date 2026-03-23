"""
Nemotron Multi-Document Reasoning — Gradio Demo
================================================
Comparative analysis of two documents using NVIDIA Nemotron 3 Super
with transparent reasoning traces. Upload or paste two documents,
select an analysis mode, and watch the model reason through differences.
"""

import os
import time
import html as html_mod
import re
import json

import gradio as gr
from openai import OpenAI

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

CLIENT = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
)

PORT = 7865

# ---------------------------------------------------------------------------
# Example Document Pairs
# ---------------------------------------------------------------------------

EXAMPLE_PAIRS = {
    "Term Sheets: Series A vs Series B": {
        "doc_a_title": "Series A Term Sheet — Meridian Ventures",
        "doc_a": """SERIES A PREFERRED STOCK TERM SHEET
Issuer: NovaBio Therapeutics, Inc.
Investor: Meridian Ventures Fund III, LP
Date: January 15, 2026

OFFERING TERMS
Security: Series A Preferred Stock
Aggregate Amount: $12,000,000
Pre-Money Valuation: $38,000,000
Price Per Share: $4.75
Shares Issued: 2,526,316

DIVIDENDS
8% cumulative, compounding annually, payable upon liquidation or redemption.

LIQUIDATION PREFERENCE
1x non-participating preferred. Series A holders receive the greater of (a) 1x their original investment plus accrued dividends or (b) their pro-rata share of remaining assets.

ANTI-DILUTION
Broad-based weighted average anti-dilution protection.

BOARD COMPOSITION
5 directors: 2 appointed by Series A holders, 2 by common holders, 1 independent mutually agreed.

PROTECTIVE PROVISIONS
Consent of majority Series A holders required for:
- Issuing senior or pari passu securities
- Amending the charter
- Declaring dividends on common stock
- Any acquisition, merger, or asset sale exceeding $500,000
- Incurring debt exceeding $250,000

INFORMATION RIGHTS
Annual audited financials, quarterly unaudited financials, monthly management reports, annual operating budget. Available to holders of at least 500,000 shares.

PRO-RATA RIGHTS
Major investors (holding >= 500,000 shares) have the right to participate in future financings to maintain ownership percentage.

RIGHT OF FIRST REFUSAL & CO-SALE
Company ROFR on founder shares. Investors have co-sale rights on any founder transfer.

VESTING
All founder shares subject to 4-year vesting with 1-year cliff. Acceleration: 50% single-trigger on change of control.

DRAG-ALONG
Holders of 60% of preferred and common (voting together) can compel all shareholders to approve a sale.

EXCLUSIVITY
45-day no-shop period from execution of this term sheet.

GOVERNING LAW
State of Delaware.""",

        "doc_b_title": "Series B Term Sheet — Polaris Growth",
        "doc_b": """SERIES B PREFERRED STOCK TERM SHEET
Issuer: NovaBio Therapeutics, Inc.
Investor: Polaris Growth Partners, LP
Date: March 8, 2026

OFFERING TERMS
Security: Series B Preferred Stock
Aggregate Amount: $35,000,000
Pre-Money Valuation: $120,000,000
Price Per Share: $11.20
Shares Issued: 3,125,000

DIVIDENDS
6% non-cumulative, payable when and if declared by the Board.

LIQUIDATION PREFERENCE
1.5x participating preferred with a 3x cap. Series B holders receive 1.5x their original investment plus accrued dividends, then participate pro-rata in remaining proceeds until aggregate returns reach 3x.

ANTI-DILUTION
Narrow-based weighted average anti-dilution protection.

BOARD COMPOSITION
7 directors: 2 appointed by Series B holders, 2 by Series A holders, 2 by common holders, 1 independent mutually agreed.

PROTECTIVE PROVISIONS
Consent of majority Series B holders required for:
- Issuing senior or pari passu securities
- Amending the charter in a way that adversely affects Series B
- Declaring dividends
- Any acquisition, merger, or asset sale
- Incurring debt exceeding $1,000,000
- Changing the size of the Board
- Hiring or terminating the CEO

INFORMATION RIGHTS
Annual audited financials, quarterly unaudited financials, monthly management reports, annual operating budget, cap table updates. Available to holders of at least 250,000 shares.

PRO-RATA RIGHTS
Major investors (holding >= 250,000 shares) have the right to participate in future financings. Super pro-rata rights (up to 2x ownership) for first $5M of any subsequent round.

PAY-TO-PLAY
Investors who do not participate in their pro-rata share of subsequent qualified financings will have their preferred stock converted to common stock.

RIGHT OF FIRST REFUSAL & CO-SALE
Company ROFR on all shareholder transfers (not just founders). Investors have co-sale rights on any transfer by holders of 1% or more.

VESTING
All unvested founder and employee shares subject to existing vesting schedules. No acceleration on change of control unless approved by Board.

DRAG-ALONG
Holders of 50% of Series B preferred (voting as a single class) can compel all shareholders to approve a sale at or above $200M.

REDEMPTION
Series B holders may request redemption after 5 years at 1.5x original purchase price plus accrued dividends, subject to legally available funds.

EXCLUSIVITY
60-day no-shop period from execution of this term sheet.

GOVERNING LAW
State of Delaware.""",
        "analysis_mode": "Comparative Risk Assessment",
    },

    "Vendor Proposals: Cloud Migration": {
        "doc_a_title": "Proposal A — Stratos Cloud Services",
        "doc_a": """CLOUD MIGRATION PROPOSAL
Vendor: Stratos Cloud Services
Client: GlobalRetail Corp
Date: February 2026

EXECUTIVE SUMMARY
Stratos proposes a 6-month phased migration of GlobalRetail's on-premise infrastructure to AWS. Total cost: $2.4M. Team of 8 engineers dedicated full-time.

SCOPE
Phase 1 (Months 1-2): Assessment and planning
- Full infrastructure audit (450+ servers, 120 databases)
- Application dependency mapping
- Migration strategy per workload (rehost, replatform, refactor)
- Security and compliance gap analysis

Phase 2 (Months 3-4): Core migration
- Database migration using AWS DMS
- Application migration using CloudEndure
- Network reconfiguration (VPC, Direct Connect)
- Identity and access management setup

Phase 3 (Months 5-6): Optimization and handoff
- Performance tuning and right-sizing
- Cost optimization (Reserved Instances, Savings Plans)
- Monitoring setup (CloudWatch, custom dashboards)
- Knowledge transfer (40 hours of training)
- 30-day hypercare support post-migration

PRICING
Fixed price: $2,400,000
Payment: 30% upfront, 40% at Phase 2 completion, 30% at final acceptance
Change orders: T&M at $275/hour

SLA
- 99.9% uptime during migration windows
- Maximum 4-hour planned downtime per application
- Data loss: zero tolerance
- Rollback capability maintained for 72 hours per migration batch

TEAM
- 1 Solutions Architect (AWS Certified)
- 1 Project Manager (PMP)
- 2 Database Migration Specialists
- 2 Infrastructure Engineers
- 1 Security Engineer
- 1 DevOps Engineer

WARRANTY
90-day warranty on all migration work. Bugs and configuration issues fixed at no additional cost.""",

        "doc_b_title": "Proposal B — Nimbus Infrastructure",
        "doc_b": """CLOUD MIGRATION PROPOSAL
Vendor: Nimbus Infrastructure Partners
Client: GlobalRetail Corp
Date: February 2026

EXECUTIVE SUMMARY
Nimbus proposes a 9-month migration of GlobalRetail's infrastructure to a multi-cloud environment (AWS primary, Azure secondary). Total cost: $3.1M. Dedicated team of 12 including 2 embedded client-side engineers.

SCOPE
Phase 1 (Months 1-3): Discovery and architecture
- Automated infrastructure discovery (all servers, databases, network topology)
- Application portfolio rationalization (retire, retain, rehost, replatform, refactor, repurchase — full 6R analysis)
- Multi-cloud architecture design with DR strategy
- Compliance framework mapping (PCI-DSS, SOC 2, GDPR)
- Total Cost of Ownership model (3-year projection)

Phase 2 (Months 4-7): Migration execution
- Database migration (DMS + custom ETL for complex schemas)
- Application migration (mix of CloudEndure, Terraform IaC, and manual refactoring)
- Multi-cloud networking (AWS VPC + Azure VNet, site-to-site VPN, ExpressRoute)
- Zero-trust security architecture implementation
- CI/CD pipeline migration and modernization

Phase 3 (Months 8-9): Optimization and managed transition
- FinOps implementation (real-time cost dashboards, automated right-sizing)
- Chaos engineering tests (Gremlin)
- Observability platform (Datadog integration)
- Knowledge transfer (80 hours of training + runbooks)
- 60-day hypercare support with 24/7 on-call

PRICING
Time & Materials with a cap: $3,100,000 (not to exceed)
Payment: Monthly invoicing based on actuals, net-30
Rate card: Junior $195/hr, Mid $250/hr, Senior $325/hr, Architect $400/hr
Savings guarantee: If actual spend is under $2.8M, Nimbus receives 20% of the savings as a performance bonus.

SLA
- 99.95% uptime during migration windows
- Maximum 2-hour planned downtime per application
- Data loss: zero tolerance with real-time replication verification
- Rollback capability maintained for 7 days per migration batch
- Incident response: 15-minute acknowledgment, 1-hour resolution for P1

TEAM
- 1 Chief Architect (AWS + Azure certified)
- 1 Program Manager (PMP, SAFe)
- 3 Cloud Engineers (2 AWS, 1 Azure)
- 2 Database Migration Specialists
- 2 Security Engineers (1 specializing in zero-trust)
- 1 DevOps/SRE Engineer
- 2 Embedded Engineers (on-site at GlobalRetail)

WARRANTY
180-day warranty. All defects fixed at no cost. Includes 2 optimization reviews at 90 and 180 days.

ADDITIONAL TERMS
- IP: All Terraform modules, scripts, and runbooks become client property
- Non-solicitation: Nimbus will not recruit GlobalRetail employees for 12 months
- Exit clause: Client may terminate with 30 days notice, paying only for completed work""",
        "analysis_mode": "Comparative Risk Assessment",
    },

    "Employment Contracts: Offer vs Counter-offer": {
        "doc_a_title": "Original Offer — Apex Robotics",
        "doc_a": """EMPLOYMENT OFFER LETTER
Company: Apex Robotics, Inc.
Position: VP of Engineering
Candidate: [Candidate Name]
Date: March 1, 2026

COMPENSATION
Base Salary: $320,000 per year
Signing Bonus: $50,000 (repayable if voluntary departure within 12 months)
Annual Bonus Target: 25% of base salary, based on company and individual performance
Equity: 150,000 stock options, 4-year vesting with 1-year cliff, strike price = FMV at grant date
Equity Refresh: Eligible for annual refresh grants starting Year 2

BENEFITS
- Health, dental, vision (company covers 90% of premiums for employee + family)
- 401(k) with 4% company match
- 20 days PTO + 10 company holidays + 5 sick days
- $5,000 annual professional development budget
- Relocation package: up to $25,000

ROLE DETAILS
- Reports to: CEO
- Direct reports: 4 engineering directors (~60 engineers total)
- Location: San Francisco HQ (hybrid — 3 days in-office)
- Start date: April 15, 2026

SEVERANCE
In the event of termination without cause: 3 months base salary continuation + 3 months COBRA coverage.

CHANGE OF CONTROL
If terminated within 12 months of a change of control: 6 months base salary + 50% accelerated vesting of unvested options.

NON-COMPETE
12-month non-compete within robotics and autonomous systems industry. Geographic scope: United States.

CONFIDENTIALITY
Standard CIIA (Confidential Information and Inventions Assignment) agreement required.

This offer expires March 15, 2026.""",

        "doc_b_title": "Counter-offer — Nexus Dynamics",
        "doc_b": """EMPLOYMENT OFFER LETTER
Company: Nexus Dynamics (Series C, $180M raised)
Position: VP of Engineering
Candidate: [Candidate Name]
Date: March 10, 2026

COMPENSATION
Base Salary: $295,000 per year
Signing Bonus: $75,000 (repayable pro-rata if voluntary departure within 18 months)
Annual Bonus Target: 30% of base salary (company performance: 70% weight, individual: 30% weight)
Equity: 0.4% of fully-diluted shares as RSUs, 4-year vesting with 1-year cliff
Equity Refresh: Guaranteed Year 2 refresh of at least 0.1% fully-diluted
Performance Bonus: Additional 0.05% equity grant if engineering OKRs achieved in Year 1

BENEFITS
- Health, dental, vision (company covers 100% of premiums for employee + family)
- 401(k) with 6% company match (immediate vesting)
- Unlimited PTO (minimum 15 days encouraged) + 12 company holidays
- $10,000 annual learning and conference budget
- Home office stipend: $3,000 one-time + $150/month
- No relocation required (fully remote with quarterly on-sites)

ROLE DETAILS
- Reports to: CTO
- Direct reports: 3 engineering directors + 2 staff engineers (~45 engineers total)
- Location: Fully remote (US-based). Quarterly on-sites in Austin, TX (travel covered)
- Start date: Flexible, targeting May 1, 2026

SEVERANCE
Termination without cause: 6 months base salary continuation + 6 months COBRA + 6 months accelerated vesting.

CHANGE OF CONTROL
If terminated within 18 months of a change of control: 12 months base salary + 100% accelerated vesting of unvested RSUs.

NON-COMPETE
None. Standard non-solicitation of employees and customers for 12 months only.

INVENTION ASSIGNMENT
Work product assignment limited to inventions directly related to company business. Prior inventions explicitly carved out.

ADVISORY
After any departure, candidate is offered a 1-year advisory role at $2,000/month with 0.02% additional equity vesting monthly.

This offer expires March 20, 2026.""",
        "analysis_mode": "Side-by-Side Clause Comparison",
    },

    "Insurance Policies: Standard vs Premium": {
        "doc_a_title": "Standard Cyber Policy — Sentinel Insurance",
        "doc_a": """CYBER LIABILITY INSURANCE POLICY
Insurer: Sentinel Insurance Group
Policy Type: Standard Cyber Liability
Policyholder: [Company Name]
Effective: April 1, 2026
Policy Number: CYB-2026-STD-4401

COVERAGE LIMITS
Aggregate Limit: $2,000,000
Per-Occurrence Limit: $1,000,000
Retention (Deductible): $25,000

INSURING AGREEMENTS

1. Data Breach Response
Coverage for reasonable and necessary costs arising from a data breach:
- Forensic investigation (up to $250,000)
- Legal counsel and regulatory compliance
- Notification costs (postal and electronic)
- Credit monitoring for affected individuals (12 months)
- Public relations and crisis management (up to $50,000)

2. Third-Party Liability
Defense and indemnity for claims arising from:
- Failure to protect personally identifiable information (PII)
- Failure to protect third-party corporate information
- Violation of privacy laws (CCPA, state privacy statutes)
Sub-limit: $1,000,000

3. Regulatory Proceedings
Defense costs and fines/penalties (where insurable by law) arising from regulatory actions related to a data breach.
Sub-limit: $500,000

4. Business Interruption
Loss of net income and extra expense due to a security event that causes system downtime.
Waiting period: 12 hours
Sub-limit: $500,000
Coverage period: 90 days from event

EXCLUSIONS
- Acts of war, terrorism, or nation-state attacks
- Unencrypted portable devices (laptops, USB drives)
- Known vulnerabilities unpatched for more than 60 days
- Contractual liability (unless arising from a covered data breach)
- Prior known events or pending litigation at policy inception
- Social engineering or voluntary wire transfers
- Infrastructure operated by third-party cloud providers (unless endorsed)

CONDITIONS
- Insured must maintain reasonable security controls (firewalls, MFA, endpoint protection)
- Insured must notify insurer within 72 hours of discovering a breach
- Insured must use panel counsel for regulatory matters
- Annual security self-assessment required for renewal""",

        "doc_b_title": "Premium Cyber Policy — Fortress Underwriters",
        "doc_b": """CYBER LIABILITY INSURANCE POLICY
Insurer: Fortress Underwriters, Ltd.
Policy Type: Premium Cyber Liability + Technology E&O
Policyholder: [Company Name]
Effective: April 1, 2026
Policy Number: FTR-2026-PRM-8820

COVERAGE LIMITS
Aggregate Limit: $10,000,000
Per-Occurrence Limit: $5,000,000
Retention (Deductible): $50,000 (reducing to $25,000 after 2 claim-free years)

INSURING AGREEMENTS

1. Data Breach Response (Full Spectrum)
Coverage for all reasonable costs arising from a data breach or suspected breach:
- Forensic investigation (no sub-limit — full policy limits apply)
- Legal counsel (choice of counsel, not limited to panel)
- Notification costs (all channels including direct outreach)
- Credit monitoring and identity theft restoration (24 months)
- Public relations and crisis management (up to $500,000)
- Call center setup and operation (up to $200,000)
- Betterment costs for security improvements post-breach (up to $150,000)

2. Third-Party Liability
Defense and indemnity for claims arising from:
- Failure to protect PII, PHI, or corporate confidential information
- Failure to prevent transmission of malware to third parties
- Violation of privacy laws (GDPR, CCPA, HIPAA, all applicable statutes)
- Media liability (defamation, copyright infringement in digital content)
- Payment card industry (PCI) fines and assessments
Full policy limits apply (no sub-limit)

3. Regulatory Proceedings
Defense costs, fines, penalties, and consumer redress funds (where insurable) arising from any regulatory action related to privacy, data protection, or cybersecurity.
Full policy limits apply
Includes coverage for GDPR Article 83 administrative fines

4. Business Interruption (Extended)
Loss of net income, extra expense, and contingent business interruption (dependent on named third-party providers including cloud infrastructure) due to a security event.
Waiting period: 8 hours
Full policy limits apply
Coverage period: 180 days from event
Includes system failure (non-malicious) trigger

5. Cyber Extortion / Ransomware
Coverage for extortion demands, ransom payments (including cryptocurrency), and associated negotiation and remediation costs.
Sub-limit: $3,000,000
Includes proactive threat containment costs

6. Social Engineering & Funds Transfer Fraud
Coverage for direct financial loss from social engineering attacks, fraudulent instructions, or invoice manipulation.
Sub-limit: $500,000
Requires dual-authorization verification process

7. Technology Errors & Omissions
Defense and indemnity for claims arising from failure of technology products or services provided by the insured to third parties.
Sub-limit: $5,000,000

EXCLUSIONS
- Acts of war (but cyberterrorism IS covered)
- Known vulnerabilities unpatched for more than 90 days (with 30-day cure period after notification)
- Intentional criminal acts by C-suite executives
- Prior known events at policy inception
- Bodily injury or property damage (except digital assets)

CONDITIONS
- Insured must maintain security controls consistent with NIST CSF or ISO 27001
- Insured must notify insurer within 48 hours of discovering a breach (or suspected breach)
- Choice of counsel permitted for claims above $500,000
- Insurer provides pre-breach services: annual penetration test, tabletop exercise, and security posture review
- Premium discount of 10% for SOC 2 Type II certified organizations
- Dedicated claims handler assigned at policy inception""",
        "analysis_mode": "Comparative Risk Assessment",
    },
}

# ---------------------------------------------------------------------------
# Analysis Prompts
# ---------------------------------------------------------------------------

ANALYSIS_MODES = {
    "Comparative Risk Assessment": {
        "system": (
            "You are an expert document analyst specializing in comparative risk assessment. "
            "You analyze two documents side by side and produce a structured, thorough comparison. "
            "For each material difference, assess the risk implications for the party receiving or signing the document. "
            "Be specific — cite exact clauses, numbers, and terms. "
            "End with a clear verdict table and recommendation."
        ),
        "prompt_template": (
            "Compare these two documents and produce a detailed comparative risk assessment.\n\n"
            "DOCUMENT A: {doc_a_title}\n"
            "---\n{doc_a}\n---\n\n"
            "DOCUMENT B: {doc_b_title}\n"
            "---\n{doc_b}\n---\n\n"
            "Structure your analysis as follows:\n"
            "1. **Executive Summary** — 2-3 sentence overview of the key differences\n"
            "2. **Clause-by-Clause Comparison** — For each material term, compare Document A vs Document B with risk implications\n"
            "3. **Risk Matrix** — Rate each document across key dimensions (Financial Risk, Operational Risk, Legal Exposure, Flexibility) on a 1-5 scale\n"
            "4. **Hidden Risks & Missing Terms** — What is absent from each document that should concern the reader?\n"
            "5. **Verdict** — Which document is more favorable to the recipient, and why?\n"
            "6. **Negotiation Priorities** — Top 3 clauses to negotiate in each document"
        ),
    },
    "Side-by-Side Clause Comparison": {
        "system": (
            "You are a meticulous contract analyst. You map every clause in Document A to its counterpart in Document B, "
            "identify additions, deletions, and modifications, and assess the practical impact of each difference. "
            "Present findings in a clear, tabular format where possible."
        ),
        "prompt_template": (
            "Perform a detailed side-by-side clause comparison of these two documents.\n\n"
            "DOCUMENT A: {doc_a_title}\n"
            "---\n{doc_a}\n---\n\n"
            "DOCUMENT B: {doc_b_title}\n"
            "---\n{doc_b}\n---\n\n"
            "For each clause or section:\n"
            "1. Show Document A's position vs Document B's position\n"
            "2. Classify the difference: **Favorable** / **Unfavorable** / **Neutral** / **New Clause** / **Removed**\n"
            "3. Explain the practical impact\n\n"
            "End with:\n"
            "- A summary count (how many favorable, unfavorable, neutral, new, removed)\n"
            "- The 3 most consequential differences\n"
            "- A recommendation on which document better protects the recipient's interests"
        ),
    },
    "Due Diligence Red Flags": {
        "system": (
            "You are a senior due diligence analyst. Your job is to find problems — ambiguities, one-sided terms, "
            "unusual provisions, missing protections, and potential deal-breakers. Be direct and flag everything "
            "that a careful reviewer should question before signing."
        ),
        "prompt_template": (
            "Review these two documents for red flags, ambiguities, and potential deal-breakers.\n\n"
            "DOCUMENT A: {doc_a_title}\n"
            "---\n{doc_a}\n---\n\n"
            "DOCUMENT B: {doc_b_title}\n"
            "---\n{doc_b}\n---\n\n"
            "For each document, identify:\n"
            "1. **Red Flags** — Terms that are unusually aggressive, one-sided, or non-market\n"
            "2. **Ambiguities** — Language that is vague or could be interpreted multiple ways\n"
            "3. **Missing Protections** — Standard terms that are absent\n"
            "4. **Deal-Breakers** — Terms that a reasonable party should refuse to accept without modification\n\n"
            "Rate each document's overall risk level: LOW / MEDIUM / HIGH / CRITICAL\n"
            "End with a combined view: which document requires more negotiation before signing?"
        ),
    },
    "Financial Impact Analysis": {
        "system": (
            "You are a financial analyst specializing in contract economics. You quantify the financial implications "
            "of every term — costs, liabilities, penalties, cap exposure, and total worst-case scenarios. "
            "Show your calculations and assumptions."
        ),
        "prompt_template": (
            "Analyze the financial implications of these two documents.\n\n"
            "DOCUMENT A: {doc_a_title}\n"
            "---\n{doc_a}\n---\n\n"
            "DOCUMENT B: {doc_b_title}\n"
            "---\n{doc_b}\n---\n\n"
            "For each document, calculate or estimate:\n"
            "1. **Total Direct Cost** — All explicit costs and payments\n"
            "2. **Contingent Liabilities** — Maximum exposure under penalty, indemnity, or liquidation clauses\n"
            "3. **Opportunity Cost** — Lock-in periods, exclusivity, non-competes valued in terms of foregone alternatives\n"
            "4. **Best Case vs Worst Case** — Financial outcome range for each document\n"
            "5. **Cost Comparison Table** — Side-by-side financial summary\n\n"
            "End with a net financial advantage calculation: which document is cheaper/more valuable overall, by how much, and under what assumptions?"
        ),
    },
}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

NVIDIA_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f0fdf0", c100="#dcfce7", c200="#bbf7d0", c300="#86efac",
        c400="#4ade80", c500="#76b900", c600="#65a300", c700="#4d7a00",
        c800="#365200", c900="#1a2900", c950="#0d1500",
    ),
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.gray,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    body_text_color="#e6edf3",
    body_text_color_dark="#e6edf3",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="#30363d",
    block_border_color_dark="#30363d",
    block_label_text_color="#8b949e",
    block_label_text_color_dark="#8b949e",
    block_title_text_color="#e6edf3",
    block_title_text_color_dark="#e6edf3",
    input_background_fill="#0d1117",
    input_background_fill_dark="#0d1117",
    input_border_color="#30363d",
    input_border_color_dark="#30363d",
    button_primary_background_fill="#76b900",
    button_primary_background_fill_dark="#76b900",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_background_fill_hover="#65a300",
    button_primary_background_fill_hover_dark="#65a300",
    button_secondary_background_fill="#21262d",
    button_secondary_background_fill_dark="#21262d",
    button_secondary_text_color="#e6edf3",
    button_secondary_text_color_dark="#e6edf3",
    button_secondary_border_color="#30363d",
    button_secondary_border_color_dark="#30363d",
)

CSS = """
.banner-container {
    text-align: center;
    padding: 20px 0 10px 0;
}
.banner-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #76b900;
    margin-bottom: 4px;
}
.banner-subtitle {
    font-size: 0.95rem;
    color: #8b949e;
}
.nvidia-badge {
    display: inline-block;
    background: linear-gradient(135deg, #76b900, #4d7a00);
    color: white;
    padding: 2px 10px;
    border-radius: 4px;
    font-weight: 700;
    font-size: 2.55rem;
    margin-right: 8px;
    vertical-align: middle;
}
.reasoning-block {
    background: #0d2818;
    border: 1px solid #1a5c2e;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.55;
    color: #76b900;
    white-space: pre-wrap;
    max-height: 500px;
    overflow-y: auto;
}
.reasoning-label {
    font-weight: 700;
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #4d7a00;
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}
.analysis-output {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #e6edf3;
}
.stats-panel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #e6edf3;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 14px;
}
.stats-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 0.85rem;
}
.stats-label { color: #8b949e; }
.stats-value { color: #e6edf3; font-family: 'JetBrains Mono', monospace; }
.doc-label {
    font-weight: 700;
    color: #76b900;
    font-size: 0.9rem;
    margin-bottom: 4px;
}
.mode-badge {
    display: inline-block;
    background: #1c2333;
    border: 1px solid #30363d;
    color: #58a6ff;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 6px;
}
footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def esc(text):
    return html_mod.escape(text)


def approx_tokens(text):
    return len(text.split()) if text else 0


def stats_html(reasoning_tokens, content_tokens, elapsed, model_name):
    return (
        f'<div class="stats-panel">'
        f'<div class="stats-row"><span class="stats-label">Model</span>'
        f'<span class="stats-value">{esc(model_name)}</span></div>'
        f'<div class="stats-row"><span class="stats-label">Reasoning</span>'
        f'<span class="stats-value">~{reasoning_tokens:,} tokens</span></div>'
        f'<div class="stats-row"><span class="stats-label">Output</span>'
        f'<span class="stats-value">~{content_tokens:,} tokens</span></div>'
        f'<div class="stats-row"><span class="stats-label">Time</span>'
        f'<span class="stats-value">{elapsed:.1f}s</span></div>'
        f'<div class="stats-row"><span class="stats-label">Reasoning</span>'
        f'<span class="stats-value" style="color:#76b900;">Enabled</span></div>'
        f'</div>'
    )


def empty_stats():
    return (
        '<div class="stats-panel">'
        '<div style="color:#8b949e; text-align:center; padding:12px;">Load an example or paste your documents, then click Analyze.</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Core Analysis — Streaming
# ---------------------------------------------------------------------------


def run_analysis(doc_a_title, doc_a, doc_b_title, doc_b, analysis_mode,
                 reasoning_budget, temperature):
    """Stream comparative analysis from Nemotron with reasoning."""

    if not doc_a.strip() or not doc_b.strip():
        yield (
            "*Please provide both documents before running analysis.*",
            "",
            empty_stats(),
        )
        return

    if not NVIDIA_API_KEY:
        yield (
            "*Error: NVIDIA_API_KEY environment variable is not set.*",
            "",
            empty_stats(),
        )
        return

    mode_config = ANALYSIS_MODES[analysis_mode]
    prompt = mode_config["prompt_template"].format(
        doc_a_title=doc_a_title or "Document A",
        doc_a=doc_a,
        doc_b_title=doc_b_title or "Document B",
        doc_b=doc_b,
    )

    messages = [
        {"role": "system", "content": mode_config["system"]},
        {"role": "user", "content": prompt},
    ]

    extra_body = {
        "chat_template_kwargs": {"enable_thinking": True},
    }

    t0 = time.time()

    try:
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=16384,
            temperature=temperature,
            top_p=0.95,
            stream=True,
            timeout=1800,
            extra_body=extra_body,
        )
    except Exception as e:
        yield (
            f"**API Error:** {esc(str(e))}",
            "",
            empty_stats(),
        )
        return

    reasoning_acc = ""
    content_acc = ""

    for chunk in response:
        for choice in chunk.choices:
            delta = choice.delta

            reasoning = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if reasoning:
                reasoning_acc += reasoning

            content = getattr(delta, "content", None)
            if content:
                content_acc += content

        elapsed = time.time() - t0
        r_tokens = approx_tokens(reasoning_acc)
        c_tokens = approx_tokens(content_acc)

        reasoning_display = ""
        if reasoning_acc:
            reasoning_display = (
                f'<div class="reasoning-block">'
                f'<div class="reasoning-label">Reasoning Trace ({r_tokens:,} tokens)</div>'
                f'{esc(reasoning_acc)}'
                f'</div>'
            )

        yield (
            content_acc,
            reasoning_display,
            stats_html(r_tokens, c_tokens, elapsed, MODEL),
        )

    # Final yield
    elapsed = time.time() - t0
    r_tokens = approx_tokens(reasoning_acc)
    c_tokens = approx_tokens(content_acc)

    reasoning_display = ""
    if reasoning_acc:
        reasoning_display = (
            f'<div class="reasoning-block">'
            f'<div class="reasoning-label">Reasoning Trace ({r_tokens:,} tokens)</div>'
            f'{esc(reasoning_acc)}'
            f'</div>'
        )

    yield (
        content_acc,
        reasoning_display,
        stats_html(r_tokens, c_tokens, elapsed, MODEL),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(title="Nemotron Multi-Document Reasoning") as demo:

        # Banner
        gr.HTML(
            '<div class="banner-container">'
            '<div class="banner-title">'
            '<span class="nvidia-badge">NVIDIA</span> Multi-Document Reasoning'
            '</div>'
            '<div class="banner-subtitle">'
            'Comparative document analysis with transparent chain-of-thought'
            '</div>'
            '</div>'
        )

        with gr.Row():
            # ---- Sidebar ----
            with gr.Column(scale=1, min_width=280):

                gr.Markdown("### Model")
                gr.HTML(
                    '<div style="font-size:0.85rem; color:#8b949e; line-height:1.8;">'
                    f'<b style="color:#e6edf3;">Endpoint</b> Nemotron Super 49B<br>'
                    '<b style="color:#e6edf3;">Reasoning</b> Always-on (chain-of-thought)<br>'
                    '<b style="color:#e6edf3;">API</b> NIM (OpenAI-compatible)<br>'
                    '<b style="color:#e6edf3;">Max Output</b> 16,384 tokens'
                    '</div>'
                )

                gr.Markdown("### Analysis Mode")
                analysis_mode = gr.Radio(
                    choices=list(ANALYSIS_MODES.keys()),
                    value="Comparative Risk Assessment",
                    label="Analysis Mode",
                    interactive=True,
                )

                gr.Markdown("### Parameters")
                reasoning_budget = gr.Slider(
                    minimum=1024,
                    maximum=16384,
                    value=8192,
                    step=512,
                    label="Reasoning Budget",
                    interactive=True,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Temperature",
                    interactive=True,
                )

                gr.Markdown("### Response Stats")
                stats_display = gr.HTML(empty_stats())

                gr.Markdown("### Load Example Pair")
                example_buttons = []
                for name in EXAMPLE_PAIRS:
                    btn = gr.Button(name, variant="secondary", size="sm")
                    example_buttons.append((btn, name))

            # ---- Main Panel ----
            with gr.Column(scale=3):

                # Document inputs — side by side
                with gr.Row():
                    with gr.Column():
                        doc_a_title = gr.Textbox(
                            label="Document A Title",
                            placeholder="e.g., Series A Term Sheet",
                            lines=1,
                        )
                        doc_a = gr.Textbox(
                            label="Document A",
                            placeholder="Paste the full text of Document A...",
                            lines=12,
                            max_lines=20,
                        )
                    with gr.Column():
                        doc_b_title = gr.Textbox(
                            label="Document B Title",
                            placeholder="e.g., Series B Term Sheet",
                            lines=1,
                        )
                        doc_b = gr.Textbox(
                            label="Document B",
                            placeholder="Paste the full text of Document B...",
                            lines=12,
                            max_lines=20,
                        )

                analyze_btn = gr.Button(
                    "Analyze Documents",
                    variant="primary",
                    size="lg",
                )

                # Output
                gr.Markdown("### Analysis")
                analysis_output = gr.Markdown(
                    value="*Load an example pair or paste your own documents, then click Analyze.*",
                )

                with gr.Accordion("Reasoning Trace", open=False):
                    reasoning_display = gr.HTML(
                        '<div class="reasoning-block">'
                        '<div class="reasoning-label">Reasoning will appear here</div>'
                        'Run an analysis to see the model\'s chain-of-thought reasoning process.'
                        '</div>'
                    )

        # ---- Example loading ----
        def load_example(name):
            pair = EXAMPLE_PAIRS[name]
            return (
                pair["doc_a_title"],
                pair["doc_a"],
                pair["doc_b_title"],
                pair["doc_b"],
                pair["analysis_mode"],
            )

        for btn, name in example_buttons:
            btn.click(
                lambda n=name: load_example(n),
                outputs=[doc_a_title, doc_a, doc_b_title, doc_b, analysis_mode],
            )

        # ---- Analysis ----
        analyze_btn.click(
            run_analysis,
            inputs=[
                doc_a_title, doc_a, doc_b_title, doc_b,
                analysis_mode, reasoning_budget, temperature,
            ],
            outputs=[analysis_output, reasoning_display, stats_display],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_port=PORT, share=False, theme=NVIDIA_THEME, css=CSS)
