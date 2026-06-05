# System Architecture

## Level 2 Data Flow Diagram (DFD)

The following sequence diagram illustrates the Dynamic Mapping Mechanism, showing how the user's interaction triggers the database pull and model inference.

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Streamlit Interface
    participant Script as Python Application Logic
    participant DB as MongoDB Atlas
    participant LLM as Google Gemini API

    User->>Frontend: 1. Selects Product (e.g. "Brand X Face Wash")
    Frontend->>Script: 2. Passes product_name
    
    User->>Frontend: 3. Clicks "Generate Fresh Analysis"
    Frontend->>Script: 4. Triggers analysis function
    
    Script->>DB: 5. Query: Find 200 reviews where Product_Name = selected_product
    DB-->>Script: 6. Returns JSON review data
    
    Script->>Script: 7. Format: Combine top 200 reviews into a single text string
    
    Script->>LLM: 8. Prompt: "Act as a product analyst..." + combined reviews
    LLM-->>Script: 9. Returns Summary, Pros, and Cons
    
    Script-->>Frontend: 10. Passes parsed analysis
    Frontend-->>User: 11. Dynamically renders insights on screen
```

## Tech Stack Overview

1. **Frontend & Router**: `Streamlit` - Handles UI, dropdowns, and rerunning the Python logic.
2. **Database**: `MongoDB Atlas` - NoSQL cloud database storing unstructured review data.
3. **The Brain**: `Google Gemini` - LLM used to process up to 1 million tokens and generate fresh insights.
4. **Hosting**: `Streamlit Community Cloud` - Directly links to GitHub for deployment (Day 7).
