"""
Eveready Product Review Sentiment Analyzer
BiLSTM + Attention Architecture
"""

import os
import re
import glob
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings('ignore')

# Global summarizer
_summarizer_tokenizer = None
_summarizer_model = None

def get_summarizer():
    global _summarizer_tokenizer, _summarizer_model
    if _summarizer_model is None:
        print("\n  Loading local abstractive summarizer (T5-Small)...")
        _summarizer_tokenizer = AutoTokenizer.from_pretrained("t5-small")
        _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return _summarizer_tokenizer, _summarizer_model

# Download NLTK data
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ============================================================
# CONFIG
# ============================================================
MAX_SEQ_LEN = 120
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 4
MIN_WORD_FREQ = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ============================================================
# STEP 1: CONSOLIDATION
# ============================================================
def consolidate_csvs(data_dir):
    """Merge all CSV files into a single DataFrame with Product_Name column."""
    print("=" * 60)
    print("STEP 1: DATA CONSOLIDATION")
    print("=" * 60)

    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if 'Body' in df.columns and 'Rating' in df.columns:
                product_name = os.path.splitext(os.path.basename(f))[0]
                df['Product_Name'] = product_name
                all_dfs.append(df)
                print(f"  Loaded {f}: {len(df)} reviews")
        except Exception as e:
            print(f"  Skipping {f}: {e}")

    master = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total reviews loaded: {len(master)}")
    return master


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
def parse_rating(rating_str):
    """Extract numeric rating from '4.0 out of 5 stars' format."""
    if pd.isna(rating_str):
        return None
    match = re.search(r'(\d+\.?\d*)', str(rating_str))
    return float(match.group(1)) if match else None

def rating_to_label(rating):
    """Convert 1-5 star rating to 3-class sentiment label."""
    if rating is None or np.isnan(rating):
        return None
    if rating <= 2.0:
        return 0  # Negative
    elif rating <= 3.0:
        return 1  # Neutral
    else:
        return 2  # Positive

def clean_data(df):
    """Clean the dataset: parse ratings, remove dupes, generate labels."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    # Combine Title + Body for richer text
    df['text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
    df['text'] = df['text'].str.strip()

    # Parse ratings
    df['rating_num'] = df['Rating'].apply(parse_rating)

    # Drop rows with no text or no rating
    before = len(df)
    df = df.dropna(subset=['text', 'rating_num'])
    df = df[df['text'].str.len() > 0]
    print(f"  Dropped {before - len(df)} rows with missing text/rating")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=['Author', 'rating_num', 'text'])
    print(f"  Dropped {before - len(df)} duplicate reviews")

    # Generate sentiment labels
    df['label'] = df['rating_num'].apply(rating_to_label)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Print class distribution
    dist = df['label'].value_counts().sort_index()
    print(f"\n  Class Distribution:")
    for lbl, count in dist.items():
        pct = count / len(df) * 100
        print(f"    {LABEL_MAP[lbl]:>10}: {count:>5} ({pct:.1f}%)")

    print(f"\n  Clean dataset size: {len(df)}")
    return df.reset_index(drop=True)


# ============================================================
# STEP 3: TEXT PREPROCESSING
# ============================================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Lowercase, remove noise, tokenize, remove stopwords, lemmatize."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)          # URLs
    text = re.sub(r'<[^>]+>', '', text)                     # HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)                    # special chars
    text = re.sub(r'\d+', '', text)                         # numbers
    text = re.sub(r'\s+', ' ', text).strip()                # extra spaces
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 1]
    return tokens

def preprocess_dataframe(df):
    """Apply preprocessing to the entire dataframe."""
    print("\n" + "=" * 60)
    print("STEP 3: TEXT PREPROCESSING")
    print("=" * 60)

    df['tokens'] = df['text'].apply(preprocess_text)
    df['clean_text'] = df['tokens'].apply(lambda t: ' '.join(t))

    avg_len = df['tokens'].apply(len).mean()
    print(f"  Average token count per review: {avg_len:.1f}")
    print(f"  Sample preprocessed text: {df['tokens'].iloc[0][:15]}...")
    return df


# ============================================================
# STEP 4: VOCABULARY & VECTORIZATION
# ============================================================
def build_vocab(token_lists, min_freq=MIN_WORD_FREQ):
    """Build vocabulary from token lists. Returns word2idx mapping."""
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.most_common():
        if freq >= min_freq:
            word2idx[word] = idx
            idx += 1

    return word2idx

def tokens_to_indices(tokens, word2idx, max_len=MAX_SEQ_LEN):
    """Convert token list to padded index array."""
    indices = [word2idx.get(t, word2idx['<UNK>']) for t in tokens[:max_len]]
    # Pad
    indices += [word2idx['<PAD>']] * (max_len - len(indices))
    return indices

def vectorize(df):
    """Build vocab and convert all tokens to index sequences."""
    print("\n" + "=" * 60)
    print("STEP 4: VOCABULARY & VECTORIZATION")
    print("=" * 60)

    word2idx = build_vocab(df['tokens'].tolist())
    print(f"  Vocabulary size: {len(word2idx)} words")

    df['indices'] = df['tokens'].apply(lambda t: tokens_to_indices(t, word2idx))
    return df, word2idx


# ============================================================
# STEP 5: PYTORCH DATASET & MODEL
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, indices, labels):
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]


class Attention(nn.Module):
    """Bahdanau-style additive attention."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.W(hidden_states))  # (batch, seq_len, hidden*2)
        scores = self.v(energy).squeeze(-1)           # (batch, seq_len)
        weights = torch.softmax(scores, dim=-1)       # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))        # (batch, seq, embed)
        lstm_out, _ = self.lstm(embedded)                  # (batch, seq, hidden*2)
        context, attn_weights = self.attention(lstm_out)   # (batch, hidden*2)
        logits = self.fc(context)                          # (batch, num_classes)
        return logits, attn_weights


# ============================================================
# STEP 6: TRAINING & EVALUATION
# ============================================================
def train_model(model, train_loader, val_loader, class_weights):
    """Train with early stopping."""
    print("\n" + "=" * 60)
    print("STEP 6: TRAINING")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        train_loss /= total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                logits, _ = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

        val_loss /= total
        val_acc = correct / total

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader):
    """Evaluate on test set and print metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits, _ = model(X)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    target_names = [LABEL_MAP[i] for i in range(NUM_CLASSES)]
    print("\n" + classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    print("  Confusion Matrix:")
    print(f"  {'':>12} {'Pred Neg':>10} {'Pred Neu':>10} {'Pred Pos':>10}")
    for i, row in enumerate(cm):
        print(f"  {target_names[i]:>12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    return all_preds, all_labels


# ============================================================
# STEP 7: PER-PRODUCT DYNAMIC SUMMARY
# ============================================================

def extract_review_phrases(review_texts, max_phrases=5):
    """
    Extract meaningful phrases directly from raw review text.
    Returns a list of short, actionable insight strings.
    
    Strategy: split reviews into sentences/clauses, score them
    by specificity (penalise generic praise/complaints), and
    return the top unique phrases.
    """
    generic_patterns = re.compile(
        r'^(good|nice|ok|okay|bad|worst|best|super|awesome|great|excellent|'
        r'terrific|fabulous|wonderful|classy|highly recommended|must buy|'
        r'just wow|perfect|fair|decent|average|not bad)[\.\!\s]*$',
        re.IGNORECASE
    )
    
    # Split each review into clauses (by period, comma, exclamation, or "but"/"and")
    clauses = []
    for text in review_texts:
        if pd.isna(text) or not isinstance(text, str):
            continue
        # Remove the leading "X.0•Title " pattern from our merged Title+Body
        text = re.sub(r'^\d+\.\d+\s*', '', text)
        # Split on bullet separator
        parts = text.split('•')
        for part in parts:
            # Further split on sentence boundaries
            sents = re.split(r'[\.!\?]+', part)
            for s in sents:
                s = s.strip()
                if len(s) > 10 and not generic_patterns.match(s):
                    clauses.append(s)
    
    # Score clauses: prefer medium-length, specific phrases
    scored = []
    seen_lower = set()
    for clause in clauses:
        lower = clause.lower().strip()
        # De-duplicate near-identical phrases
        if lower in seen_lower:
            continue
        # Skip if too short or too long
        word_count = len(lower.split())
        if word_count < 3 or word_count > 20:
            continue
        # Penalise very generic text
        generic_word_count = len(re.findall(
            r'\b(good|nice|ok|okay|bad|worst|best|super|awesome|great|excellent)\b',
            lower
        ))
        specificity = word_count - (generic_word_count * 2)
        if specificity < 2:
            continue
        seen_lower.add(lower)
        scored.append((clause, specificity))
    
    # Sort by specificity (most specific first) and return top N
    scored.sort(key=lambda x: x[1], reverse=True)
    # Capitalize first letter and clean up each phrase
    results = []
    for phrase, _ in scored[:max_phrases]:
        phrase = phrase.strip().rstrip(',')
        # Strip emojis and non-ASCII to avoid Windows encoding issues
        phrase = phrase.encode('ascii', 'ignore').decode('ascii').strip()
        if not phrase:
            continue
        phrase = phrase[0].upper() + phrase[1:]
        results.append(phrase)
    return results


def generate_improvement_suggestions(neg_phrases):
    """
    Convert raw negative review phrases into actionable improvement
    suggestions using keyword pattern matching.
    """
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

def is_spam_review(text):
    """
    Detect repetitive/spam reviews like "good product good product good product".
    Returns True if the review appears to be spammy keyword repetition.
    """
    if pd.isna(text) or not isinstance(text, str):
        return True
        
    text = text.lower().strip()
    words = text.split()
    total_words = len(words)
    
    if total_words < 4:
        return False
        
    unique_words = len(set(words))
    
    # If the ratio of unique words to total words is very low, it's highly repetitive.
    # E.g. "good product good product good product" -> 2 unique / 6 total = 0.33
    if total_words > 4 and (unique_words / total_words) < 0.4:
        return True
        
    # Check if a single word appears too many times and dominates the text
    if total_words > 5:
        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count > 4 and (most_common_count / total_words) > 0.4:
            return True
            
    return False


def generate_product_summaries(df, model, word2idx):
    """Generate a dynamic sentiment summary for each product."""
    print("\n" + "=" * 60)
    print("STEP 7: PRODUCT SENTIMENT SUMMARIES")
    print("=" * 60)

    model.eval()
    products = df['Product_Name'].unique()

    # Get predictions for all data
    all_indices = np.array(df['indices'].tolist())
    dataset = ReviewDataset(all_indices, np.zeros(len(df), dtype=int))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(DEVICE)
            logits, _ = model(X)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)

    df['predicted_sentiment'] = [LABEL_MAP[p] for p in all_preds]

    catalog_data = []

    print("\n" + "-" * 60)
    for product in sorted(products):
        pdf = df[df['Product_Name'] == product]
        total = len(pdf)
        pos = (pdf['predicted_sentiment'] == 'Positive').sum()
        neu = (pdf['predicted_sentiment'] == 'Neutral').sum()
        neg = (pdf['predicted_sentiment'] == 'Negative').sum()

        pos_pct = pos / total * 100
        neu_pct = neu / total * 100
        neg_pct = neg / total * 100

        # Determine overall sentiment
        if pos_pct >= 60:
            overall = "Highly Positive"
        elif pos_pct >= 40:
            overall = "Generally Positive"
        elif neg_pct >= 40:
            overall = "Generally Negative"
        else:
            overall = "Mixed"

        # --- Extract REAL phrases from positive and negative reviews (filtering spam) ---
        pos_reviews = [r for r in pdf[pdf['predicted_sentiment'] == 'Positive']['text'].tolist() if not is_spam_review(r)]
        neg_reviews = [r for r in pdf[pdf['predicted_sentiment'] == 'Negative']['text'].tolist() if not is_spam_review(r)]

        pos_phrases = extract_review_phrases(pos_reviews, max_phrases=20)
        neg_phrases = extract_review_phrases(neg_reviews, max_phrases=5)

        # Generate actionable improvement suggestions from negative phrases
        improvements = generate_improvement_suggestions(neg_phrases)
        # If pattern matching didn't catch everything, use the raw phrases
        if not improvements and neg_phrases:
            improvements = neg_phrases[:3]

        # Get average rating
        avg_rating = pdf['rating_num'].mean()

        print(f"\n  Product: {product}")
        print(f"  Reviews: {total} | Avg Rating: {avg_rating:.1f}/5")
        print(f"  Sentiment: {overall}")
        print(f"    Positive: {pos:>4} ({pos_pct:5.1f}%) {'#' * int(pos_pct // 2)}")
        print(f"    Neutral:  {neu:>4} ({neu_pct:5.1f}%) {'#' * int(neu_pct // 2)}")
        print(f"    Negative: {neg:>4} ({neg_pct:5.1f}%) {'#' * int(neg_pct // 2)}")

        if pos_phrases:
            print(f"  Strengths (from reviews):")
            for p in pos_phrases:
                print(f"    + {p}")
        if improvements:
            print(f"  Areas to Improve:")
            for imp in improvements:
                print(f"    - {imp}")
        if neg_phrases:
            print(f"  Customer Complaints (verbatim):")
            for p in neg_phrases:
                print(f"    ! \"{p}\"")

        # Generate abstractive summary using local model
        text_to_summarize = f"This product has {pos_pct:.0f}% positive reviews. "
        if pos_phrases:
            text_to_summarize += "Customers heavily praise the following aspects: " + ". ".join(pos_phrases[:12]) + ". "
        if improvements:
            text_to_summarize += "However, critical areas to improve are: " + ". ".join(improvements[:8]) + ". "
        elif neg_phrases:
            text_to_summarize += "However, some commonly complain that: " + ". ".join(neg_phrases[:8]) + ". "

        try:
            tokenizer, summarizer_model = get_summarizer()
            # T5 is typically used with 'summarize: ' prefix
            inputs = tokenizer("summarize: " + text_to_summarize, return_tensors="pt", max_length=1024, truncation=True)
            outputs = summarizer_model.generate(inputs.input_ids, max_length=220, min_length=140, do_sample=False)
            summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Basic cleanup: capitalize first letter
            if summary_text:
                summary_text = summary_text[0].upper() + summary_text[1:]
        except Exception as e:
            print(f"    Summarization failed: {e}")
            summary_text = f"Reviewers were {pos_pct:.0f}% positive about this product."

        print(f"  Summary: {summary_text}")

        # Extract raw reviews to include in JSON
        raw_reviews = []
        for _, row in pdf.iterrows():
            raw_reviews.append({
                "author": row['Author'] if pd.notna(row['Author']) else "Unknown",
                "rating": row['rating_num'],
                "title": row['Title'] if pd.notna(row['Title']) else "",
                "body": row['Body'] if pd.notna(row['Body']) else "",
                "sentiment": row['predicted_sentiment']
            })
            
        # Extract Aspect-based Categorization
        ASPECT_PATTERNS = {
            'Battery / Power': r'\b(batter(y|ies)|back.?up|charg(e|ing|er|es)|power|drain|discharge|cell(s)?)\b',
            'Brightness': r'\b(bright(ness)?|dim|glow|light(s)?|lumen(s)?|beam|focus|illuminate|intensity)\b',
            'Value for Money': r'\b(pric(e|es|y)|value|money|cost|cheap|expens(ive)?|worth|budget|affordable|overpriced)\b',
            'Build Quality': r'\b(build|sturd(y|iness)|broke(n)?|break|crack(ed)?|plastic|heavy|weight|design|finish|material|feel|fragile)\b',
            'Reliability': r'\b(stop(ped)?|work(s|ing|ed)?|defect(ive|s)?|fault(y|s)?|last(s|ed)?|life(span)?|durable|fail(ed|ure)?)\b',
            'Usability': r'\b(use(d|s|ful)?|function(s|al)?|easy|switch(es)?|button(s)?|handle(s)?|grip|handy|operate|control|convenient|ergonomic)\b',
            'Size & Portability': r'\b(size|small|big|large|compact|portable|pocket|fit(s)?|carry|bulky|dimension(s)?)\b',
            'Packaging & Delivery': r'\b(packag(e|ing|ed)|deliver(y|ed|s)|box(es)?|seal(ed)?|damage(d)?|transit|shipping|courier)\b',
            'Appearance': r'\b(look(s|ing|ed)?|color(s)?|colour(s)?|ugly|beautiful|attractive|aesthetic|style(s)?)\b'
        }

        aspects_data = []
        for aspect_name, pattern in ASPECT_PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            matching_reviews = []
            
            for _, row in pdf.iterrows():
                # Clean up any '5.0•Title' or '5.0 Title' prefix formatting
                text = str(row['text'])
                text = re.sub(r'^\d+\.\d+[\s•]+', '', text).strip()
                
                if regex.search(text) and not is_spam_review(text):
                    matching_reviews.append({
                        "body": text,
                        "sentiment": row['predicted_sentiment']
                    })
            
            count = len(matching_reviews)
            if count > 0:
                pos_count = sum(1 for r in matching_reviews if r['sentiment'] == 'Positive')
                neg_count = sum(1 for r in matching_reviews if r['sentiment'] == 'Negative')
                
                # Generate specific short summary for this aspect
                aspect_text_to_summarize = f"Reviews regarding {aspect_name}: "
                aspect_phrases = []
                for r in matching_reviews[:20]: # Pull context from top matches
                    sentences = re.split(r'[\.!\?]+', r['body'])
                    for s in sentences:
                        s_lower = s.lower()
                        if regex.search(s_lower) and len(s.split()) > 3:
                            # User requested to remove generic 'excellent' phrases
                            if "excellent" in s_lower:
                                continue
                            # Filter other overly generic phrases
                            generic_count = len(re.findall(r'\b(good|nice|ok|okay|bad|worst|best|super|awesome|great)\b', s_lower))
                            specificity = len(s.split()) - (generic_count * 2)
                            if specificity > 2:
                                aspect_phrases.append(s.strip())
                                break
                
                aspect_text_to_summarize += ". ".join(aspect_phrases[:8])
                
                try:
                    tokenizer, summarizer_model = get_summarizer()
                    inputs = tokenizer("summarize: " + aspect_text_to_summarize, return_tensors="pt", max_length=512, truncation=True)
                    outputs = summarizer_model.generate(inputs.input_ids, max_length=60, min_length=15, do_sample=False)
                    aspect_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if aspect_summary:
                        aspect_summary = aspect_summary[0].upper() + aspect_summary[1:]
                except Exception as e:
                    aspect_summary = f"Customers mentioned {aspect_name} in {count} reviews."
                
                # Sort reviews by length so we don't send massive paragraphs for quotes
                sorted_quotes = sorted([r['body'] for r in matching_reviews], key=len)
                
                aspects_data.append({
                    "name": aspect_name,
                    "count": count,
                    "pos": pos_count,
                    "neg": neg_count,
                    "summary": aspect_summary,
                    "quotes": sorted_quotes[:5] # Top 5 shortest quotes for the UI snippet
                })

        # Sort aspects by frequency
        aspects_data.sort(key=lambda x: x['count'], reverse=True)

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
            "aspects": aspects_data,
            "raw_reviews": raw_reviews,
            "image": f"images/{product}.jpg"
        })

    # Save to catalog_data.js
    data_dir = os.path.dirname(os.path.abspath(__file__))
    js_path = os.path.join(data_dir, 'catalog_data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write("const catalogData = ")
        json.dump(catalog_data, f, indent=4)
        f.write(";\n")
    print(f"\n  Saved {len(catalog_data)} products to catalog_data.js")

    print("\n" + "-" * 60)

    # Overall brand summary
    total_all = len(df)
    pos_all = (df['predicted_sentiment'] == 'Positive').sum()
    neg_all = (df['predicted_sentiment'] == 'Negative').sum()
    print(f"\n  OVERALL EVEREADY BRAND SENTIMENT")
    print(f"  Total Reviews Analyzed: {total_all}")
    print(f"  Brand Positivity Rate: {pos_all/total_all*100:.1f}%")
    print(f"  Brand Negativity Rate: {neg_all/total_all*100:.1f}%")

    avg_rat = df['rating_num'].mean()
    print(f"  Average Star Rating: {avg_rat:.2f}/5")

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1
    df = consolidate_csvs(data_dir)

    # Step 2
    df = clean_data(df)

    # Step 3
    df = preprocess_dataframe(df)

    # Step 4
    df, word2idx = vectorize(df)

    # Save master CSV
    master_path = os.path.join(data_dir, 'master_reviews.csv')
    df[['Product_Name', 'Author', 'rating_num', 'label', 'text', 'clean_text']].to_csv(
        master_path, index=False)
    print(f"\n  Saved master_reviews.csv ({len(df)} rows)")

    # Prepare tensors
    X = np.array(df['indices'].tolist())
    y = np.array(df['label'].tolist())

    # Train/Val/Test split (80/10/10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\n  Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # DataLoaders
    train_ds = ReviewDataset(X_train, y_train)
    val_ds = ReviewDataset(X_val, y_val)
    test_ds = ReviewDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Compute class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    class_weights = torch.tensor(len(y_train) / (NUM_CLASSES * class_counts), dtype=torch.float32)
    print(f"  Class weights: {class_weights.tolist()}")

    # Step 5 - Build model
    print("\n" + "=" * 60)
    print("STEP 5: MODEL ARCHITECTURE")
    print("=" * 60)
    vocab_size = len(word2idx)
    model = BiLSTMAttention(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: BiLSTM + Attention")
    print(f"  Vocab: {vocab_size}, Embed: {EMBED_DIM}, Hidden: {HIDDEN_DIM}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Device: {DEVICE}")

    # Step 6 - Train
    model = train_model(model, train_loader, val_loader, class_weights)

    # Evaluate
    evaluate_model(model, test_loader)

    # Step 7 - Product summaries
    df = generate_product_summaries(df, model, word2idx)

    # Save model
    model_path = os.path.join(data_dir, 'sentiment_model.pt')
    torch.save({
        'model_state': model.state_dict(),
        'word2idx': word2idx,
        'config': {
            'vocab_size': vocab_size, 'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES
        }
    }, model_path)
    print(f"\n  Model saved to {model_path}")
    print("\n  PIPELINE COMPLETE!")


if __name__ == '__main__':
    main()
