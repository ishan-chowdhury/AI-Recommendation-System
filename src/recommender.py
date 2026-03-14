from datetime import datetime, timedelta


class RecommendationEngine:
    """
    Analyzes categorized transactions and generates actionable saving tips.
    Tip priority: Frequency → Budget Overrun → High Percentage (>40%) → Budgeting Goal → Dominant (>50%) → Default.
    """

    def __init__(self, monthly_budget: float = 10000):
        # Per-category spending thresholds (₹) – can be personalized later
        self.thresholds = {
            "Food and Drink": 2000,
            "Entertainment": 1500,
            "Travel": 3000,
            "Shopping": 2500,
            "Transportation": 1000,
            "Health and Fitness": 500,
            "Utilities": 1000,
            "Education": 2000,
            "Personal Care": 500,
            "Other": 1000,
        }

        # Fixed monthly budget goal for the budgeting tip
        self.monthly_budget = monthly_budget

    def generate_tips(self, transactions: list) -> str:
        """
        Analyzes a list of transactions to provide a saving tip.
        Input format: List of dicts with {'category': str, 'amount': float, 'name': str}
        """
        if not transactions:
            return "Start adding expenses to get personalized AI insights!"

        # ── 1. Aggregate spending by category ──────────────────────────
        category_totals: dict[str, float] = {}
        item_frequency: dict[str, int] = {}
        total_spent: float = 0

        for tx in transactions:
            cat = tx.get("category")
            amt = tx.get("amount", 0)
            name = tx.get("name", "").lower()

            category_totals[cat] = category_totals.get(cat, 0) + amt
            item_frequency[name] = item_frequency.get(name, 0) + 1
            total_spent += amt

        # ── 2. Frequency Analysis (The "Small Leak" Rule) ─────────────
        # Items bought ≥ 3 times
        frequent_items = [name for name, count in item_frequency.items() if count >= 3]
        if frequent_items:
            item = frequent_items[0].title()
            return (
                f"You've purchased {item} {item_frequency[frequent_items[0]]} times recently. "
                f"Small daily expenses add up -- try cutting back next week!"
            )

        # ── 3. Budget Overruns per category ───────────────────────────
        for cat, limit in self.thresholds.items():
            if category_totals.get(cat, 0) > limit:
                excess = category_totals[cat] - limit
                return (
                    f"Your {cat} spending is Rs.{excess:,.0f} over your typical limit. "
                    f"Consider skipping non-essential {cat} purchases for a few days."
                )

        # ── 4. High Percentage Warning (>40% of total) ───────────────
        if total_spent > 0:
            for cat, amt in category_totals.items():
                pct = (amt / total_spent) * 100
                if pct > 40:
                    return (
                        f"[WARNING] {cat} takes up {pct:.0f}% of your total spending (Rs.{amt:,.0f} / Rs.{total_spent:,.0f}). "
                        f"Try to diversify your expenses to avoid over-concentration in one area."
                    )

        # ── 5. Budgeting Tip (vs. monthly goal) ──────────────────────
        if total_spent > self.monthly_budget:
            over = total_spent - self.monthly_budget
            return (
                f"[BUDGET] You've spent Rs.{total_spent:,.0f} against your Rs.{self.monthly_budget:,.0f} monthly goal - "
                f"that's Rs.{over:,.0f} over budget. Review non-essential categories to get back on track."
            )
        else:
            remaining = self.monthly_budget - total_spent
            return (
                f"[OK] Great job! You've spent Rs.{total_spent:,.0f} so far, "
                f"Rs.{remaining:,.0f} under your Rs.{self.monthly_budget:,.0f} monthly goal. Keep it up!"
            )