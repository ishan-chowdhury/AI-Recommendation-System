# AI Recommendation System

A standalone ML microservice for a **Money Management System** that automatically categorizes expenses and generates personalized spending tips. Built with a **HybridClassifier** (Fuzzy Matching + Zero-Shot NLP) and a **RecommendationEngine**.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Self-Learning](#self-learning)
- [Integration with Next.js](#integration-with-nextjs)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Features

- **Hybrid Classification**: Combines fast fuzzy string matching with a Zero-Shot NLP model (`facebook/bart-large-mnli`) for accurate expense categorization
- **Input Normalization**: Cleans messy input (extra spaces, special characters, mixed case) before matching
- **Confidence Threshold**: Defaults to "Other" if the ML model's confidence is below 30%, preventing bad categorizations
- **Self-Learning**: Automatically learns new items from ML classifications and saves them to disk for faster future matching
- **Smart Recommendations**: Generates spending tips based on frequency analysis, budget overruns, category concentration, and monthly budget goals
- **FastAPI REST API**: Production-ready endpoints with Pydantic request validation
- **Bulk Classification**: Classify multiple items in a single request to reduce latency

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Next.js Frontend                    │
│              (Receipt Scanner / Manual Input)         │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (JSON)
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Server (main.py)                 │
│                                                      │
│  POST /classify        → Single item classification  │
│  POST /bulk-classify   → Batch classification        │
│  POST /recommend       → Spending tips               │
│  GET  /health          → Health check                │
└────────┬─────────────────────────┬──────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ HybridClassifier │    │ RecommendationEngine │
│  (classifier.py) │    │   (recommender.py)   │
│                  │    │                      │
│ 1. Normalize     │    │ 1. Frequency check   │
│ 2. Fuzzy Match   │    │ 2. Budget overruns   │
│ 3. Zero-Shot ML  │    │ 3. >40% warning      │
│ 4. Auto-Learn    │    │ 4. Monthly goal      │
└────────┬─────────┘    └──────────────────────┘
         │
         ▼
┌──────────────────┐
│ learned_items.json│  (auto-generated persistence)
└──────────────────┘
```

---

## Project Structure

```
AI-Recommendation-System/
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── classifier.py        # HybridClassifier (Fuzzy + Zero-Shot ML)
│   ├── recommender.py       # RecommendationEngine (spending tips)
│   ├── main.py              # FastAPI app with all endpoints
│   └── test_standalone.py   # Standalone test script
├── data/
│   ├── learned_items.json   # Auto-generated: items learned by ML
│   └── mock_data.json       # Placeholder for mock data
├── .env                     # IDE configuration (PYTHONPATH)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Install Dependencies

```bash
cd AI-Recommendation-System
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |
| `transformers` | HuggingFace Zero-Shot NLP model |
| `torch` | PyTorch backend for transformers |
| `pydantic` | Request/response validation |
| `thefuzz` | Fuzzy string matching |
| `python-Levenshtein` | C accelerator for thefuzz (~10x faster) |

> **Note**: On first run, the `facebook/bart-large-mnli` model (~1.6 GB) will be downloaded and cached by HuggingFace. Subsequent runs will load from cache.

---

## Running the Service

### Start the API Server

From the project root:

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Or from inside the `src/` folder:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Once running, visit:
- **API**: `http://localhost:8000`
- **Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Run the Standalone Test

```bash
python src/test_standalone.py
```

This runs 5 test steps covering normalization, classification, high-percentage warnings, budgeting tips, and self-learning.

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok", "service": "AI Recommendation System" }
```

---

### `POST /classify`

Classify a single item/merchant name.

**Request:**
```json
{ "name": "Zomato - Biryani" }
```

**Response:**
```json
{
  "item": "Zomato - Biryani",
  "category": "Food and Drink",
  "confidence": 0.92,
  "method": "fuzzy_matching"
}
```

---

### `POST /bulk-classify`

Classify multiple items in one request (reduces network round-trips).

**Request:**
```json
{
  "items": ["Zomato - Biryani", "Uber Ride", "Netflix"]
}
```

**Response:**
```json
[
  { "item": "Zomato - Biryani", "category": "Food and Drink", "confidence": 0.92, "method": "fuzzy_matching" },
  { "item": "Uber Ride", "category": "Transportation", "confidence": 0.87, "method": "zero_shot_ml" },
  { "item": "Netflix", "category": "Entertainment", "confidence": 0.90, "method": "fuzzy_matching" }
]
```

---

### `POST /recommend`

Generate a spending tip from categorized transactions.

**Request:**
```json
{
  "transactions": [
    { "name": "Zomato", "amount": 450, "category": "Food and Drink" },
    { "name": "Uber Ride", "amount": 200, "category": "Transportation" },
    { "name": "PVR Cinemas", "amount": 1200, "category": "Entertainment" }
  ]
}
```

**Response:**
```json
{
  "tip": "[OK] Great job! You've spent Rs.1,850 so far, Rs.8,150 under your Rs.10,000 monthly goal. Keep it up!"
}
```

---

## How It Works

### Classification Pipeline

When an item name (e.g., `"  Zomato - BIRYANI!!  "`) is sent to `/classify`:

1. **Normalize**: `"  Zomato - BIRYANI!!  "` → `"zomato biryani"` (lowercase, remove special chars, collapse spaces)
2. **Fuzzy Match**: Compare against 80+ known Indian brands/merchants. If match score > 85% → return immediately
3. **ML Fallback**: If no fuzzy match, use `facebook/bart-large-mnli` zero-shot classification against 10 categories
4. **Confidence Check**: If ML confidence < 30% → default to `"Other"`
5. **Auto-Learn**: If ML confidence >= 30% → save the item to `learned_items.json` for future fuzzy matching

### Spending Categories

| Category | Examples |
|----------|----------|
| Food and Drink | Zomato, Swiggy, McDonald's, BigBasket |
| Shopping | Flipkart, Myntra, Amazon, Croma |
| Transportation | Ola, Uber, Metro, Auto Rickshaw |
| Health and Fitness | Apollo Pharmacy, Cult.fit, 1mg |
| Entertainment | Netflix, PVR, Spotify, BookMyShow |
| Utilities | Jio, Airtel, Tata Power, Gas |
| Education | Byju's, Unacademy, Udemy |
| Travel | MakeMyTrip, Indigo, Oyo |
| Personal Care | Lakme, Urban Company, Nykaa |
| Other | Anything below 30% ML confidence |

### Recommendation Engine

Tips are generated in priority order (first match wins):

| Priority | Rule | Example Tip |
|----------|------|------------|
| 1 | Item bought >= 3 times | "You've purchased Zomato 5 times recently..." |
| 2 | Category exceeds threshold | "Your Food spending is Rs.500 over your limit..." |
| 3 | Category > 40% of total | "[WARNING] Food takes up 65% of total spending..." |
| 4 | Total vs. monthly budget | "[BUDGET] You've spent Rs.12,000 against Rs.10,000 goal..." |

---

## Self-Learning

The classifier improves over time:

1. When an item can't be fuzzy-matched, the ML model classifies it
2. If the ML confidence is >= 30%, the item and its category are saved to `data/learned_items.json`
3. On the next startup, these learned items are loaded into the knowledge base
4. Future requests for the same item will be matched via fast fuzzy matching instead of the slower ML model

**Example** `data/learned_items.json`:
```json
{
  "Uber Ride": "Transportation",
  "PVR Cinemas": "Entertainment"
}
```

---

## Integration with Next.js

This service is designed to run alongside your Next.js money management app. Example integration in a Next.js API route:

```typescript
// app/api/bill/process/route.ts
export async function POST(request: Request) {
  const { items, amounts } = await request.json();

  // Step 1: Classify all items in one request
  const classifyRes = await fetch("http://localhost:8000/bulk-classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  const classified = await classifyRes.json();

  // Step 2: Build transactions with amounts + categories
  const transactions = classified.map((c, i) => ({
    name: c.item,
    amount: amounts[i],
    category: c.category,
  }));

  // Step 3: Get spending tip
  const tipRes = await fetch("http://localhost:8000/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transactions }),
  });
  const { tip } = await tipRes.json();

  return Response.json({ transactions, tip });
}
```

---

## Configuration

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| Monthly Budget | `RecommendationEngine(monthly_budget=...)` | Rs.10,000 | Budget goal for the budgeting tip |
| ML Confidence Threshold | `classifier.py` line 38 | 0.3 (30%) | Below this, category defaults to "Other" |
| Fuzzy Match Threshold | `classifier.py` line 225 | 85 | Fuzzy score must exceed this to match |
| Category Thresholds | `recommender.py` lines 12-23 | Varies | Per-category spending limits |
| Server Port | uvicorn CLI `--port` | 8000 | API server port |

---

## Testing

### Run the full test suite:
```bash
python src/test_standalone.py
```

### Test individual endpoints (with server running):

```bash
# Health check
curl http://localhost:8000/health

# Classify
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"name": "Zomato"}'

# Bulk classify
curl -X POST http://localhost:8000/bulk-classify \
  -H "Content-Type: application/json" \
  -d '{"items": ["Zomato", "Uber", "Netflix"]}'

# Recommend
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"name": "Zomato", "amount": 450, "category": "Food and Drink"}]}'
```

Or visit `http://localhost:8000/docs` for the interactive Swagger UI.
