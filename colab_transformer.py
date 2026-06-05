# ============================================================
# EVEREADY SENTIMENT ANALYZER — GOOGLE COLAB VERSION
# DistilBERT Sentiment + Flan-T5 Summarization
# ============================================================
# HOW TO USE:
# 1. Open Google Colab: https://colab.research.google.com
# 2. Click Runtime -> Change runtime type -> Select GPU (T4)
# 3. Upload your master_reviews.csv when prompted
# 4. Run all cells
# 5. Download the generated catalog_data.js at the end
# ============================================================

# --- CELL 1: Install dependencies ---
# !pip install transformers sentencepiece torch pandas scikit-learn -q

# --- CELL 2: Upload your data ---
from google.colab import files
import io

print("Please upload your master_reviews.csv file...")
uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]
print(f"Uploaded: {csv_filename}")

# --- CELL 3: Full Pipeline ---
import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
MAX_SEQ_LEN = 128
NUM_CLASSES = 3
BATCH_SIZE = 32         # Colab can handle larger batches
EPOCHS = 3
LR = 2e-5
PATIENCE = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ============================================================
# DATASET CLASS
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)

df = pd.read_csv(csv_filename)
print(f"  Loaded {len(df)} reviews")

df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

dist = df['label'].value_counts().sort_index()
print(f"\n  Class Distribution:")
for lbl, count in dist.items():
    pct = count / len(df) * 100
    print(f"    {LABEL_MAP[lbl]:>10}: {count:>5} ({pct:.1f}%)")

# ============================================================
# STEP 2: TOKENIZE WITH DISTILBERT
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DISTILBERT TOKENIZATION")
print("=" * 60)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

texts = df['text'].tolist()
labels = df['label'].tolist()

texts_train, texts_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
texts_val, texts_test, y_val, y_test = train_test_split(
    texts_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Split: Train={len(texts_train)}, Val={len(texts_val)}, Test={len(texts_test)}")

train_enc = tokenizer(texts_train, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='pt')
val_enc = tokenizer(texts_val, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='pt')
test_enc = tokenizer(texts_test, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='pt')

train_ds = ReviewDataset(train_enc, y_train)
val_ds = ReviewDataset(val_enc, y_val)
test_ds = ReviewDataset(test_enc, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
class_weights = torch.tensor(len(y_train) / (NUM_CLASSES * class_counts), dtype=torch.float32)
print(f"  Class weights: {class_weights.tolist()}")

# ============================================================
# STEP 3: FINE-TUNE DISTILBERT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FINE-TUNING DISTILBERT")
print("=" * 60)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=NUM_CLASSES
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Device: {DEVICE}")

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels_batch = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item() * len(labels_batch)
        correct += (outputs.logits.argmax(1) == labels_batch).sum().item()
        total += len(labels_batch)

    train_loss /= total
    train_acc = correct / total

    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels_batch = batch['labels'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels_batch)
            val_loss += loss.item() * len(labels_batch)
            correct += (outputs.logits.argmax(1) == labels_batch).sum().item()
            total += len(labels_batch)

    val_loss /= total
    val_acc = correct / total

    print(f"  Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_state)

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].numpy())

target_names = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
print("\n" + classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

# ============================================================
# STEP 4: LOAD FLAN-T5-BASE FOR SUMMARIZATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: LOADING FLAN-T5-BASE FOR SUMMARIZATION")
print("=" * 60)

t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base').to(DEVICE)
t5_model.eval()
print(f"  T5-base loaded on {DEVICE}!")


def generate_enhanced_template(product_name, pos_pct, neg_pct, pos_phrases, neg_phrases, improvements, avg_rating):
    """Fallback: generate a polished summary using templates."""
    parts = []
    if pos_pct >= 80:
        parts.append(f"Customers are highly enthusiastic about the {product_name}, "
                     f"with an impressive {pos_pct:.0f}% positive sentiment and a {avg_rating}/5 average rating.")
    elif pos_pct >= 60:
        parts.append(f"The {product_name} enjoys solid customer approval at {pos_pct:.0f}% positive, "
                     f"earning a {avg_rating}/5 average rating.")
    elif pos_pct >= 40:
        parts.append(f"Customer opinions on the {product_name} are mixed, "
                     f"with {pos_pct:.0f}% positive reviews and a {avg_rating}/5 average rating.")
    else:
        parts.append(f"The {product_name} faces significant customer dissatisfaction, "
                     f"with only {pos_pct:.0f}% positive reviews and a {avg_rating}/5 rating.")
    if pos_phrases:
        phrase = pos_phrases[0][:80] + '...' if len(pos_phrases[0]) > 80 else pos_phrases[0]
        parts.append(f'Buyers frequently praise: "{phrase}".')
    if neg_pct >= 20 and improvements:
        parts.append(f"However, {neg_pct:.0f}% of users report issues. "
                     f"Key areas for improvement: {'; '.join(improvements[:2])}.")
    elif neg_pct >= 10 and improvements:
        parts.append(f"A small minority ({neg_pct:.0f}%) noted concerns: {improvements[0].lower()}.")
    return ' '.join(parts)


def generate_t5_summary(product_name, pos_pct, neg_pct, pos_phrases, neg_phrases, improvements, avg_rating):
    """Use Flan-T5-base to generate a natural language product summary with quality fallback."""
    strengths_text = "; ".join(pos_phrases[:3]) if pos_phrases else "No specific strengths mentioned"
    complaints_text = "; ".join(neg_phrases[:3]) if neg_phrases else "No complaints reported"
    improvements_text = "; ".join(improvements[:3]) if improvements else "None identified"

    prompt = (
        f"You are a product analyst. Write exactly 3 sentences summarizing customer reviews for the Eveready {product_name}.\n\n"
        f"Data:\n"
        f"- Average rating: {avg_rating} out of 5 stars\n"
        f"- {pos_pct:.0f}% positive reviews, {neg_pct:.0f}% negative reviews\n"
        f"- What customers love: {strengths_text}\n"
        f"- What customers complain about: {complaints_text}\n"
        f"- Suggested improvements: {improvements_text}\n\n"
        f"Write a 3-sentence summary covering: (1) overall reception, (2) what buyers love, (3) what needs improvement. "
        f"Use specific details from the data above."
    )

    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=40,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )
    summary = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.encode('ascii', 'ignore').decode('ascii').strip()

    # Quality check: if T5 produces junk (too short or generic), fall back to template
    if len(summary.split()) < 15 or len(summary) < 60:
        print(f"    [T5 output too short: '{summary}'] -> Using enhanced template")
        summary = generate_enhanced_template(
            product_name, pos_pct, neg_pct, pos_phrases, neg_phrases, improvements, avg_rating
        )

    return summary


# ============================================================
# PHRASE EXTRACTION (Preserved from original)
# ============================================================
def extract_review_phrases(review_texts, max_phrases=5):
    generic_patterns = re.compile(
        r'^(good|nice|ok|okay|bad|worst|best|super|awesome|great|excellent|'
        r'terrific|fabulous|wonderful|classy|highly recommended|must buy|'
        r'just wow|perfect|fair|decent|average|not bad)[\.!\s]*$',
        re.IGNORECASE
    )
    clauses = []
    for text in review_texts:
        if pd.isna(text) or not isinstance(text, str):
            continue
        text = re.sub(r'^\d+\.\d+\s*', '', text)
        parts = text.split('\u2022')
        for part in parts:
            sents = re.split(r'[\.!\?]+', part)
            for s in sents:
                s = s.strip()
                if len(s) > 10 and not generic_patterns.match(s):
                    clauses.append(s)

    scored = []
    seen_lower = set()
    for clause in clauses:
        lower = clause.lower().strip()
        if lower in seen_lower:
            continue
        word_count = len(lower.split())
        if word_count < 3 or word_count > 20:
            continue
        generic_word_count = len(re.findall(
            r'\b(good|nice|ok|okay|bad|worst|best|super|awesome|great|excellent)\b', lower))
        specificity = word_count - (generic_word_count * 2)
        if specificity < 2:
            continue
        seen_lower.add(lower)
        scored.append((clause, specificity))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for phrase, _ in scored[:max_phrases]:
        phrase = phrase.strip().rstrip(',')
        phrase = phrase.encode('ascii', 'ignore').decode('ascii').strip()
        if not phrase:
            continue
        phrase = phrase[0].upper() + phrase[1:]
        results.append(phrase)
    return results


def generate_improvement_suggestions(neg_phrases):
    suggestions = []
    patterns = {
        r'batter(y|ies)': 'Battery performance needs improvement',
        r'backup|back.?up': 'Battery backup duration should be increased',
        r'stop(ped)?\s+work': 'Product durability/longevity needs improvement',
        r'broke|broken|crack': 'Build quality and sturdiness should be improved',
        r'dim|bright(ness)?|glow': 'Brightness level could be enhanced',
        r'switch|button': 'Switch/button mechanism needs to be more reliable',
        r'charg(e|ing|er)': 'Charging system needs improvement',
        r'heat|hot|overheat': 'Overheating issue needs to be addressed',
        r'leak|leaking': 'Leakage issues need to be resolved',
        r'noise|noisy|sound': 'Noise levels should be reduced',
        r'packag(e|ing)': 'Packaging quality should be improved',
        r'deliver|damaged|dent': 'Delivery handling and packaging need improvement',
        r'pric(e|y)|expens|cost': 'Price-to-value ratio could be better',
        r'life|lifespan|last': 'Product lifespan needs to be extended',
        r'defect|fault|issue': 'Quality control should be tightened',
        r'size|small|big|large|heavy|weight': 'Product size/weight could be optimized',
        r'led|bulb|light': 'Light quality/output needs attention',
        r'water|rain|wet': 'Water resistance should be improved',
    }
    matched_keys = set()
    for phrase in neg_phrases:
        phrase_lower = phrase.lower()
        for pattern, suggestion in patterns.items():
            if suggestion not in matched_keys and re.search(pattern, phrase_lower):
                suggestions.append(suggestion)
                matched_keys.add(suggestion)
                break
    return suggestions


# ============================================================
# STEP 5: GENERATE PRODUCT SUMMARIES
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: PRODUCT SENTIMENT SUMMARIES")
print("=" * 60)

# Get predictions for ALL reviews
model.eval()
all_texts = df['text'].tolist()
all_preds_full = []

for i in range(0, len(all_texts), BATCH_SIZE):
    batch_texts = all_texts[i:i+BATCH_SIZE]
    enc = tokenizer(batch_texts, max_length=MAX_SEQ_LEN, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=enc['input_ids'].to(DEVICE), attention_mask=enc['attention_mask'].to(DEVICE))
        preds = outputs.logits.argmax(1).cpu().numpy()
        all_preds_full.extend(preds)

df['predicted_sentiment'] = [LABEL_MAP[p] for p in all_preds_full]

catalog_data = []
products = df['Product_Name'].unique()

for product in sorted(products):
    pdf = df[df['Product_Name'] == product]
    total = len(pdf)
    pos = (pdf['predicted_sentiment'] == 'Positive').sum()
    neu = (pdf['predicted_sentiment'] == 'Neutral').sum()
    neg = (pdf['predicted_sentiment'] == 'Negative').sum()

    pos_pct = pos / total * 100
    neu_pct = neu / total * 100
    neg_pct = neg / total * 100

    if pos_pct >= 60:
        overall = "Highly Positive"
    elif pos_pct >= 40:
        overall = "Generally Positive"
    elif neg_pct >= 40:
        overall = "Generally Negative"
    else:
        overall = "Mixed"

    pos_reviews = pdf[pdf['predicted_sentiment'] == 'Positive']['text'].tolist()
    neg_reviews = pdf[pdf['predicted_sentiment'] == 'Negative']['text'].tolist()

    pos_phrases = extract_review_phrases(pos_reviews, max_phrases=5)
    neg_phrases = extract_review_phrases(neg_reviews, max_phrases=5)
    improvements = generate_improvement_suggestions(neg_phrases)
    if not improvements and neg_phrases:
        improvements = neg_phrases[:3]

    avg_rating = pdf['rating_num'].mean()

    # Use T5 for summary
    summary_text = generate_t5_summary(
        product, pos_pct, neg_pct,
        pos_phrases, neg_phrases, improvements,
        round(avg_rating, 1)
    )

    print(f"\n  Product: {product}")
    print(f"  Reviews: {total} | Avg Rating: {avg_rating:.1f}/5 | Sentiment: {overall}")
    print(f"    Pos: {pos_pct:.1f}% | Neu: {neu_pct:.1f}% | Neg: {neg_pct:.1f}%")
    print(f"  Summary: {summary_text}")

    catalog_data.append({
        "id": product,
        "name": product,
        "reviews": total,
        "avg_rating": round(avg_rating, 1),
        "sentiment": overall,
        "pos_pct": round(pos_pct, 1),
        "neu_pct": round(neu_pct, 1),
        "neg_pct": round(neg_pct, 1),
        "strengths": pos_phrases,
        "improvements": improvements,
        "complaints": neg_phrases,
        "summary": summary_text,
        "image": f"images/{product}.jpg"
    })

# ============================================================
# STEP 6: SAVE & DOWNLOAD
# ============================================================
# Save as JS file
with open('catalog_data.js', 'w') as f:
    f.write("const catalogData = ")
    json.dump(catalog_data, f, indent=4)
    f.write(";")

# Also save as JSON for the API
with open('catalog_data.json', 'w') as f:
    json.dump(catalog_data, f, indent=4)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)

# Brand summary
total_all = len(df)
pos_all = (df['predicted_sentiment'] == 'Positive').sum()
neg_all = (df['predicted_sentiment'] == 'Negative').sum()
print(f"\n  Total Reviews: {total_all}")
print(f"  Brand Positivity: {pos_all/total_all*100:.1f}%")
print(f"  Brand Negativity: {neg_all/total_all*100:.1f}%")
print(f"  Avg Rating: {df['rating_num'].mean():.2f}/5")

# Auto-download the file
print("\n  Downloading catalog_data.js to your computer...")
files.download('catalog_data.js')
print("  Done! Place this file in your eveready project folder.")
