
import re
import os
import json
from thefuzz import process 
from transformers import pipeline

# Path to the learned items JSON file (relative to the project root)
LEARNED_ITEMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'learned_items.json'
)

class HybridClassifer:
    def __init__(self):
        # 1. Initialize the Zero-Shot Classifier (NLP Model)
        print("Loading ML model... this may take a moment.")
        self.ml_classifier = pipeline(
            "zero-shot-classification", 
            model="cross-encoder/nli-MiniLM2-L6-H768",
            framework="pt"
        )
        
        # 2. Defined standard categories
        self.categories = [
            "Food and Drink",
            "Shopping",
            "Transportation",
            "Health and Fitness",
            "Entertainment",
            "Utilities",
            "Education",
            "Travel",
            "Personal Care",
            "Other"
        ]

        # 3. Confidence threshold for ML fallback
        self.ml_confidence_threshold = 0.3

        # 4. Knowledge Base for Fuzzy Matching (Indian Context)
        self.knowledge_base = {
            # --- Food and Drink (Fast Food & Delivery) ---
            "Zomato": "Food and Drink",
            "Swiggy": "Food and Drink",
            "Haldiram's": "Food and Drink",
            "Bikanervala": "Food and Drink",
            "Domino's": "Food and Drink",
            "KFC India": "Food and Drink",
            "Pizza Hut": "Food and Drink",
            "Subway": "Food and Drink",
            "McDonald's": "Food and Drink",
            "Burger Singh": "Food and Drink",
            "Wow! Momo": "Food and Drink",
            "Faasos": "Food and Drink",
            "Behrouz Biryani": "Food and Drink",
            "Chai Point": "Food and Drink",
            "Chaayos": "Food and Drink",
            "Cafe Coffee Day": "Food and Drink",
            "Starbucks": "Food and Drink",
            "Amul": "Food and Drink",
            "Mother Dairy": "Food and Drink",
            "Blinkit": "Food and Drink",
            "Zepto": "Food and Drink",
            "BigBasket": "Food and Drink",

            # --- Shopping (Retail & E-commerce) ---
            "Flipkart": "Shopping",
            "Myntra": "Shopping",
            "Ajio": "Shopping",
            "Nykaa": "Shopping",
            "Snapdeal": "Shopping",
            "Reliance Digital": "Shopping",
            "Croma": "Shopping",
            "Vijay Sales": "Shopping",
            "Tanishq": "Shopping",
            "Titan": "Shopping",
            "Westside": "Shopping",
            "Pantaloons": "Shopping",
            "Max Fashion": "Shopping",
            "Shoppers Stop": "Shopping",
            "Vishal Mega Mart": "Shopping",
            "Decathlon": "Shopping",

            # --- Transportation (Cabs & Travel) ---
            "Ola": "Transportation",
            "Uber India": "Transportation",
            "Rapido": "Transportation",
            "BluSmart": "Transportation",
            "RedBus": "Transportation",
            "IRCTC": "Transportation",
            "Delhi Metro": "Transportation",
            "Namma Metro": "Transportation",
            "Auto Rickshaw": "Transportation",

            # --- Health and Fitness ---
            "Apollo Pharmacy": "Health and Fitness",
            "Netmeds": "Health and Fitness",
            "Tata 1mg": "Health and Fitness",
            "PharmEasy": "Health and Fitness",
            "Cult.fit": "Health and Fitness",
            "MedPlus": "Health and Fitness",
            "Dr Lal PathLabs": "Health and Fitness",
            "Metropolis": "Health and Fitness",
            "Anytime Fitness": "Health and Fitness",

            # --- Entertainment (Streaming & Cinema) ---
            "Hotstar": "Entertainment",
            "Disney+ Hotstar": "Entertainment",
            "JioCinema": "Entertainment",
            "Zee5": "Entertainment",
            "SonyLIV": "Entertainment",
            "Prime Video": "Entertainment",
            "Netflix": "Entertainment",
            "BookMyShow": "Entertainment",
            "PVR Inox": "Entertainment",
            "Spotify": "Entertainment",
            "YouTube": "Entertainment",

            # --- Utilities (Telecom & Power) ---
            "Jio": "Utilities",
            "Airtel": "Utilities",
            "Vi": "Utilities",
            "BSNL": "Utilities",
            "Tata Power": "Utilities",
            "Adani Electricity": "Utilities",
            "BESCOM": "Utilities",
            "MGL": "Utilities",
            "Indane": "Utilities",
            "HP Gas": "Utilities",
            "Bharat Gas": "Utilities",

            # --- Education ---
            "Byju's": "Education",
            "Unacademy": "Education",
            "Physics Wallah": "Education",
            "Vedantu": "Education",
            "Allen": "Education",
            "Akash": "Education",
            "Udemy": "Education",

            # --- Travel ---
            "MakeMyTrip": "Travel",
            "Goibibo": "Travel",
            "Cleartrip": "Travel",
            "EaseMyTrip": "Travel",
            "Oyo": "Travel",
            "Taj Hotels": "Travel",
            "Indigo": "Travel",
            "Air India": "Travel",
            "Vistara": "Travel",

            # --- Personal Care ---
            "Lakme": "Personal Care",
            "Sugar Cosmetics": "Personal Care",
            "Mamaearth": "Personal Care",
            "Kaya": "Personal Care",
            "Enrich Salon": "Personal Care",
            "Urban Company": "Personal Care",
        }

        # 5. Load any previously learned items from disk
        self._load_learned_items()

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize input text:
        - Convert to lowercase
        - Remove special characters (keep alphanumerics and spaces)
        - Collapse multiple spaces into one
        - Strip leading/trailing whitespace
        """
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_learned_items(self):
        """Load previously learned items from the JSON file and merge into knowledge_base."""
        if os.path.exists(LEARNED_ITEMS_PATH):
            try:
                with open(LEARNED_ITEMS_PATH, 'r', encoding='utf-8') as f:
                    learned: dict = json.load(f)
                self.knowledge_base.update(learned)
                print(f"Loaded {len(learned)} learned items from disk.")
            except (json.JSONDecodeError, IOError):
                print("Warning: Could not read learned_items.json - starting fresh.")

    def _save_learned_items(self, item_name: str, category: str):
        """Persist a single new mapping to the learned-items JSON file."""
        learned = {}
        if os.path.exists(LEARNED_ITEMS_PATH):
            try:
                with open(LEARNED_ITEMS_PATH, 'r', encoding='utf-8') as f:
                    learned = json.load(f)
            except (json.JSONDecodeError, IOError):
                learned = {}

        learned[item_name] = category

        os.makedirs(os.path.dirname(LEARNED_ITEMS_PATH), exist_ok=True)
        with open(LEARNED_ITEMS_PATH, 'w', encoding='utf-8') as f:
            json.dump(learned, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def classify(self, item_name: str) -> dict:
        """
        Combined classification logic.
        1. Normalize the input string.
        2. Try fuzzy matching against the knowledge_base.
        3. Fall back to Zero-Shot ML with a confidence threshold.
        4. Auto-learn new items that the ML classifies with sufficient confidence.
        """
        normalized = self._normalize(item_name)

        # 1. Fuzzy Matching Check (against normalized KB keys)
        match, score = process.extractOne(normalized, self.knowledge_base.keys())
        if score > 85:
            return {
                "item": item_name,
                "category": self.knowledge_base[match],
                "confidence": score / 100,
                "method": "fuzzy_matching"
            }

        # 2. ML Model Fallback
        ml_result = self.ml_classifier(normalized, self.categories)
        top_label = ml_result['labels'][0]
        top_score = round(ml_result['scores'][0], 4)

        # 3. Confidence threshold – default to "Other" if too low
        if top_score < self.ml_confidence_threshold:
            return {
                "item": item_name,
                "category": "Other",
                "confidence": top_score,
                "method": "zero_shot_ml (low confidence => Other)"
            }

        # 4. Auto-learn: persist the new item for future fuzzy matching
        self.learn(item_name, top_label)

        return {
            "item": item_name,
            "category": top_label,
            "confidence": top_score,
            "method": "zero_shot_ml"
        }

    def learn(self, item_name: str, category: str):
        """
        Teach the classifier a new item → category mapping.
        Updates the in-memory knowledge_base AND persists to disk.
        """
        if item_name not in self.knowledge_base:
            self.knowledge_base[item_name] = category
            self._save_learned_items(item_name, category)
            print(f"[Learn] Added '{item_name}' => '{category}' to knowledge base.")
