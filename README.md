# Customer RFM Segmentation Analysis

> Python (pandas) RFM model that scores customers on Recency, Frequency and Monetary value and groups them into actionable behavioural segments.

## Business problem
Marketing spend is wasted when every customer gets the same message. The business needs to know who its best customers are, who is drifting away, and where retention effort pays the most.

## What's inside
| File | Purpose |
|---|---|
| `rfm_segmentation.py` | Cleans transactions, computes RFM, scores, segments, summarises, charts |
| `output/` | `rfm_customers.csv`, `rfm_segment_summary.csv`, `rfm_segments.png` |

## Quick start
```bash
pip install pandas numpy matplotlib
python rfm_segmentation.py                    # synthetic sample data
python rfm_segmentation.py --input sales.csv  # columns: customer_id, order_id, order_date, amount
```

## Method
| Metric | Definition | Scoring |
|---|---|---|
| **Recency (R)** | Days since last purchase (snapshot = last date + 1) | 5 = most recent |
| **Frequency (F)** | Number of distinct orders | 5 = most orders |
| **Monetary (M)** | Total spend | 5 = highest spend |

Scores use quintiles (1-5). Ranking with `method="first"` avoids duplicate bin-edge errors that are common with frequency data.

## Segment rules (first match wins)
| Segment | Rule | Recommended action |
|---|---|---|
| Champions | R ≥ 4 and F ≥ 4 | Reward, early access, referrals |
| Loyal | R ≥ 3 and F ≥ 3 | Upsell, loyalty programme |
| New Customers | R ≥ 4 and F ≤ 2 | Onboarding, second-purchase offer |
| At-Risk | R ≤ 2 and F ≥ 3 | Win-back campaign |
| Hibernating | R ≤ 2 and F ≤ 2 | Low-cost reactivation |
| Needs Attention | Everything else | Limited-time offers |

## Insights to look for
- What share of revenue comes from Champions and Loyal customers (revenue concentration)?
- How large and how valuable is the At-Risk group? These customers used to buy often, so they are the best win-back target.
- Are new customers converting to a second purchase?

> Add your real results here after running on actual or public transaction data (for example the UCI Online Retail dataset).

## Limitations & extensions
- Quintile cut-offs are relative to your own customer base; revisit them as the base grows.
- Extend with K-Means clustering on log-scaled RFM values and compare with the rule-based segments.
- Add customer lifetime value and churn probability.

## Author
Abdimanar Edin Guyo: Accountant, Financial Analyst & Data Analyst
