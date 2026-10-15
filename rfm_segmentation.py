"""
Project 4: Customer RFM Segmentation Analysis
----------------------------------------------
Calculates Recency, Frequency and Monetary scores from transaction data and
assigns each customer to a behavioural tier (Champions, Loyal, At-Risk, ...).

Usage
-----
    python rfm_segmentation.py                       # synthetic sample data
    python rfm_segmentation.py --input sales.csv     # your own transactions

Expected CSV columns:
    customer_id, order_id, order_date, amount

Outputs (folder `output/`):
    rfm_customers.csv   one row per customer with R/F/M values, scores, segment
    rfm_segment_summary.csv   and  rfm_segments.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output")
SEED = 7


# --------------------------------------------------------------------------
# 1. Sample transactions
# --------------------------------------------------------------------------
def generate_sample_data(n_customers: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    start = pd.Timestamp("2025-01-01")
    for cid in range(1, n_customers + 1):
        n_orders = max(1, int(rng.gamma(1.5, 3)))
        active_days = int(rng.integers(30, 600))          # customers go quiet at different times
        offsets = rng.integers(0, active_days, n_orders)
        spend_level = rng.lognormal(mean=4.0, sigma=0.6)  # customer-level spend tendency
        for k, off in enumerate(offsets):
            rows.append({
                "customer_id": f"C{cid:04d}",
                "order_id": f"O{cid:04d}-{k}",
                "order_date": start + pd.Timedelta(days=int(off)),
                "amount": round(float(spend_level * rng.uniform(0.6, 1.5)), 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["customer_id", "order_id", "order_date", "amount"])
    df = df.drop_duplicates(subset="order_id")
    df = df[df["amount"] > 0]  # remove refunds/zero-value rows
    return df


# --------------------------------------------------------------------------
# 3. RFM calculation
# --------------------------------------------------------------------------
def score_quintiles(series: pd.Series, ascending: bool) -> pd.Series:
    """Score 1-5 using quintiles. rank(method='first') avoids duplicate-edge errors."""
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def assign_segment(row: pd.Series) -> str:
    """Rule-based segmentation. Order matters: the first matching rule wins."""
    r, f = row["R"], row["F"]
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r <= 2 and f >= 3:
        return "At-Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Needs Attention"


def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    # Snapshot = day after the last transaction, so the latest buyer has Recency = 1
    snapshot_date = snapshot_date or df["order_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency=("order_date", lambda d: (snapshot_date - d.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("amount", "sum"),
    ).reset_index()

    rfm["R"] = score_quintiles(rfm["Recency"], ascending=False)    # recent buyers score 5
    rfm["F"] = score_quintiles(rfm["Frequency"], ascending=True)
    rfm["M"] = score_quintiles(rfm["Monetary"], ascending=True)
    rfm["RFM_Score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


# --------------------------------------------------------------------------
# 4. Summary + chart
# --------------------------------------------------------------------------
ACTIONS = {
    "Champions": "Reward, early access, ask for referrals",
    "Loyal": "Upsell, loyalty programme",
    "New Customers": "Onboarding series, second-purchase offer",
    "At-Risk": "Win-back campaign, personalised outreach",
    "Hibernating": "Low-cost reactivation or let go",
    "Needs Attention": "Limited-time offers, re-engage",
}


def summarise(rfm: pd.DataFrame) -> pd.DataFrame:
    s = rfm.groupby("Segment").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("Recency", "mean"),
        avg_frequency=("Frequency", "mean"),
        avg_monetary=("Monetary", "mean"),
        total_revenue=("Monetary", "sum"),
    ).round(1)
    s["pct_customers"] = (s["customers"] / s["customers"].sum() * 100).round(1)
    s["pct_revenue"] = (s["total_revenue"] / s["total_revenue"].sum() * 100).round(1)
    s["recommended_action"] = s.index.map(ACTIONS)
    return s.sort_values("total_revenue", ascending=False)


def plot_segments(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    summary["customers"].plot.barh(ax=axes[0], color="#60a5fa")
    axes[0].set(title="Customers per segment", xlabel="Customers", ylabel="")
    summary["pct_revenue"].plot.barh(ax=axes[1], color="#2dd4bf")
    axes[1].set(title="Share of revenue by segment (%)", xlabel="% of revenue", ylabel="")
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rfm_segments.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RFM segmentation")
    parser.add_argument("--input", help="CSV with customer_id, order_id, order_date, amount")
    args = parser.parse_args()

    raw = pd.read_csv(args.input) if args.input else generate_sample_data()
    tx = clean_transactions(raw)
    print(f"Transactions: {len(tx):,} | Customers: {tx['customer_id'].nunique():,}")

    rfm = build_rfm(tx)
    summary = summarise(rfm)

    OUTPUT_DIR.mkdir(exist_ok=True)
    rfm.to_csv(OUTPUT_DIR / "rfm_customers.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "rfm_segment_summary.csv")
    plot_segments(summary)

    pd.set_option("display.width", 220, "display.max_columns", None)
    print("\n=== SEGMENT SUMMARY ===")
    print(summary)
    print(f"\nSaved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
