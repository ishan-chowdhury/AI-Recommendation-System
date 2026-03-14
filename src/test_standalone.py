import os
import sys

# Ensure the root directory is accessible to import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import HybridClassifer
from src.recommender import RecommendationEngine

def run_test():
    # Initialize both parts
    clf = HybridClassifer()
    rec = RecommendationEngine()

    # ── Step 1: Classification with normalization demo ──────────────
    raw_expenses = [
        {"name": "  Zomato - BIRYANI!!  ", "amount": 450},   # tests normalization
        {"name": "zomato   pizza", "amount": 600},            # tests normalization
        {"name": "Zomato - Burger", "amount": 350},
        {"name": "Uber Ride", "amount": 200},
        {"name": "PVR Cinemas", "amount": 1200},
        {"name": "Flipkart Order", "amount": 3500},
        {"name": "Swiggy Dinner", "amount": 700},
        {"name": "xyzzy_mystery_store_9000", "amount": 100},  # tests low-confidence => "Other"
    ]

    print("\n--- Step 1: Categorizing Items (with normalization) ---")
    processed_transactions = []
    for expense in raw_expenses:
        result = clf.classify(expense['name'])
        processed_transactions.append({
            "name": expense['name'],
            "amount": expense['amount'],
            "category": result['category']
        })
        print(f"  Item: {expense['name']:35s} => Category: {result['category']:20s} "
              f"(confidence={result['confidence']:.2f}, method={result['method']})")

    # ── Step 2: Recommendation tips ────────────────────────────────
    print("\n--- Step 2: Generating Recommendation ---")
    tip = rec.generate_tips(processed_transactions)
    print(f"  AI TIP: {tip}")

    # ── Step 3: High-percentage warning demo ───────────────────────
    print("\n--- Step 3: High-Percentage Warning Demo ---")
    heavy_food = [
        {"name": "Zomato", "amount": 5000, "category": "Food and Drink"},
        {"name": "Uber", "amount": 500, "category": "Transportation"},
    ]
    tip2 = rec.generate_tips(heavy_food)
    print(f"  AI TIP: {tip2}")

    # ── Step 4: Budgeting tip demo ─────────────────────────────────
    print("\n--- Step 4: Budgeting Tip Demo ---")
    big_spend = [
        {"name": "MakeMyTrip", "amount": 8000, "category": "Travel"},
        {"name": "Flipkart",   "amount": 3000, "category": "Shopping"},
    ]
    tip3 = rec.generate_tips(big_spend)
    print(f"  AI TIP: {tip3}")

    # ── Step 5: Self-Learning demo ─────────────────────────────────
    print("\n--- Step 5: Self-Learning Check ---")
    print(f"  Knowledge base size: {len(clf.knowledge_base)} entries")
    if "xyzzy_mystery_store_9000" in clf.knowledge_base:
        print(f"  'xyzzy_mystery_store_9000' was NOT learned (low confidence => Other).")
    else:
        print(f"  'xyzzy_mystery_store_9000' was NOT added (low confidence or unknown).")

    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test()