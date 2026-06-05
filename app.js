// Global function to open reviews modal
window.openReviewsModal = function(productId) {
    if (typeof catalogData === 'undefined') return;
    const product = catalogData.find(p => p.id === productId);
    if (!product || !product.raw_reviews) return;

    const modal = document.getElementById('reviewsModal');
    const title = document.getElementById('modalProductTitle');
    const list = document.getElementById('modalReviewsList');

    title.textContent = `All Reviews: ${product.name} (${product.raw_reviews.length})`;
    list.innerHTML = '';

    product.raw_reviews.forEach(review => {
        let sentimentClass = 'neu';
        if (review.sentiment === 'Positive') sentimentClass = 'pos';
        if (review.sentiment === 'Negative') sentimentClass = 'neg';

        const card = document.createElement('div');
        card.className = `raw-review-card ${sentimentClass}`;
        
        let titleHtml = review.title ? `<div class="raw-review-title">${review.title}</div>` : '';
        
        card.innerHTML = `
            <div class="raw-review-header">
                <span class="raw-review-author">${review.author}</span>
                <span class="raw-review-rating">&#9733; ${review.rating} / 5.0</span>
            </div>
            ${titleHtml}
            <div class="raw-review-body">${review.body}</div>
        `;
        list.appendChild(card);
    });

    modal.style.display = 'block';
};

document.addEventListener('DOMContentLoaded', () => {
    // Modal Setup
    const modal = document.getElementById('reviewsModal');
    const closeBtn = document.querySelector('.close-modal');

    if (closeBtn && modal) {
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        }
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    }
    const grid = document.getElementById('productGrid');
    const dropdown = document.getElementById('productDropdown');
    
    try {
        if (typeof catalogData === 'undefined') {
            throw new Error("catalogData is not defined. Make sure catalog_data.js is loaded.");
        }
        
        const products = catalogData;
        
        // Populate dropdown
        products.forEach(product => {
            const option = document.createElement('option');
            option.value = product.id;
            option.textContent = product.name;
            dropdown.appendChild(option);
        });

        // Function to render a single product
        const renderProduct = (productId) => {
            if (!productId) {
                grid.innerHTML = '<div class="loading">Please select a product from the dropdown above.</div>';
                return;
            }

            const product = products.find(p => p.id === productId);
            if (!product) return;

            const pos = product.pos_pct;
            const neu = product.neu_pct;
            const neg = product.neg_pct;
            
            let sentimentColor = 'var(--pos-color)';
            if (neg > 40) sentimentColor = 'var(--neg-color)';
            else if (neu > pos && neu > neg) sentimentColor = 'var(--neu-color)';

            // Build strengths list
            let displayStrengths = [];
            if (product.strengths && product.strengths.length > 0) {
                const shuffled = [...product.strengths].sort(() => 0.5 - Math.random());
                displayStrengths = shuffled.slice(0, 5);
            }

            const strengthsHtml = displayStrengths.length > 0
                ? `<div class="insight-section">
                     <div class="insight-header insight-pos">&#10003; Customer Strengths</div>
                     <ul class="insight-list">
                       ${displayStrengths.map(s => `<li class="insight-item insight-item-pos">${s}</li>`).join('')}
                     </ul>
                   </div>`
                : '';

            // Build improvements list
            const improvementsHtml = product.improvements && product.improvements.length > 0
                ? `<div class="insight-section">
                     <div class="insight-header insight-neg">&#9888; Areas to Improve</div>
                     <ul class="insight-list">
                       ${product.improvements.map(s => `<li class="insight-item insight-item-neg">${s}</li>`).join('')}
                     </ul>
                   </div>`
                : '';

            // Build complaints list (verbatim quotes)
            const complaintsHtml = product.complaints && product.complaints.length > 0
                ? `<div class="insight-section">
                     <div class="insight-header insight-quote">&#128172; Customer Complaints (verbatim)</div>
                     <ul class="insight-list">
                       ${product.complaints.map(s => `<li class="insight-item insight-item-quote">"${s}"</li>`).join('')}
                     </ul>
                   </div>`
                : '';
            
            // Build Amazon-style Aspect Pills and Details Card
            let aspectsHtml = '';
            if (product.aspects && product.aspects.length > 0) {
                let pillsHtml = product.aspects.map((aspect, idx) => {
                    let icon = '&#126;'; // Neutral tilde
                    let sentimentClass = 'neu';
                    if (aspect.pos > aspect.neg * 1.5) { icon = '&#8599;'; sentimentClass = 'pos'; } // Arrow up-right
                    else if (aspect.neg > aspect.pos) { icon = '&#8600;'; sentimentClass = 'neg'; } // Arrow down-right

                    return `<button class="aspect-pill ${sentimentClass}" onclick="showAspectDetails('${product.id}', ${idx}, this)">
                                <span class="aspect-pill-icon">${icon}</span> 
                                ${aspect.name} (${aspect.count})
                            </button>`;
                }).join('');

                aspectsHtml = `
                    <div class="aspect-section-title">Customers say</div>
                    <div class="summary-text" style="border:none; padding-top:0; margin-top:0;">${product.summary}</div>
                    
                    <div class="aspect-section-title" style="font-size: 0.9rem; margin-top:1rem;">Select to learn more</div>
                    <div class="aspect-pills-container">
                        ${pillsHtml}
                    </div>
                    <div id="aspectDetailsCard_${product.id}" class="aspect-details-card">
                        <span class="close-aspect-card" onclick="this.parentElement.classList.remove('open')">&times;</span>
                        <div class="aspect-card-header" id="aspectHeader_${product.id}"></div>
                        <div class="aspect-card-stats" id="aspectStats_${product.id}"></div>
                        <div class="aspect-card-summary" id="aspectSummary_${product.id}"></div>
                        <div class="aspect-quotes-list" id="aspectQuotes_${product.id}"></div>
                    </div>
                `;
            } else {
                // Fallback if no aspects
                aspectsHtml = `<div class="summary-text">${product.summary}</div>`;
            }

            const card = document.createElement('div');
            card.className = 'product-card';
            card.style.width = '100%';
            card.style.cursor = 'default';
            card.style.transform = 'none';
            card.style.boxShadow = '0 15px 30px rgba(0,0,0,0.3)';
            
            card.innerHTML = `
                <div class="card-image">
                    <img src="${product.image}" alt="${product.name}" onerror="this.onerror=null; this.parentNode.innerHTML='<div class=\\'no-image\\'>Image pending...</div>';">
                    <div class="rating-badge">&#9733; ${product.avg_rating}</div>
                </div>
                <div class="card-content">
                    <h3 class="product-name">${product.name}</h3>
                    <div class="review-count">${product.reviews} reviews analyzed</div>
                    
                    <div class="sentiment-container">
                        <div class="sentiment-label">
                            <span style="color: ${sentimentColor}">${product.sentiment}</span>
                            <span style="color: var(--text-muted)">${pos}% Pos</span>
                        </div>
                        <div class="sentiment-bar">
                            <div class="bar-pos" style="width: ${pos}%" title="${pos}% Positive"></div>
                            <div class="bar-neu" style="width: ${neu}%" title="${neu}% Neutral"></div>
                            <div class="bar-neg" style="width: ${neg}%" title="${neg}% Negative"></div>
                        </div>
                    </div>

                    ${aspectsHtml}
                    
                    <div class="insights-toggle" onclick="this.parentElement.querySelector('.insights-panel').classList.toggle('open'); this.classList.toggle('open');">
                        View Detailed Insights <span class="toggle-arrow">&#9660;</span>
                    </div>
                    
                    <div class="insights-panel">
                        ${strengthsHtml}
                        ${improvementsHtml}
                        ${complaintsHtml}
                    </div>
                    
                    <button class="view-reviews-btn" onclick="openReviewsModal('${product.id}')">
                        View All ${product.raw_reviews ? product.raw_reviews.length : product.reviews} Reviews
                    </button>
                </div>
            `;
            
            grid.innerHTML = '';
            grid.appendChild(card);
        };

        // Global function to show aspect details
        window.showAspectDetails = function(productId, aspectIdx, btnElement) {
            const product = catalogData.find(p => p.id === productId);
            if (!product) return;
            const aspect = product.aspects[aspectIdx];
            if (!aspect) return;

            // Highlight active pill
            const container = btnElement.parentElement;
            container.querySelectorAll('.aspect-pill').forEach(btn => btn.classList.remove('active'));
            btnElement.classList.add('active');

            const card = document.getElementById(`aspectDetailsCard_${productId}`);
            const header = document.getElementById(`aspectHeader_${productId}`);
            const stats = document.getElementById(`aspectStats_${productId}`);
            const summary = document.getElementById(`aspectSummary_${productId}`);
            const quotesList = document.getElementById(`aspectQuotes_${productId}`);

            header.textContent = `${aspect.count} customers mention "${aspect.name}"`;
            stats.innerHTML = `<span class="pos">${aspect.pos} positive</span> <span class="neg">${aspect.neg} negative</span>`;
            summary.textContent = aspect.summary;
            
            if (aspect.quotes && aspect.quotes.length > 0) {
                quotesList.innerHTML = aspect.quotes.map(q => `<div class="aspect-quote-item">"${q}"</div>`).join('');
                quotesList.style.display = 'block';
            } else {
                quotesList.style.display = 'none';
            }

            card.classList.add('open');
        };

        dropdown.addEventListener('change', (e) => {
            renderProduct(e.target.value);
        });
        
    } catch (error) {
        console.error("Error loading catalog data:", error);
        grid.innerHTML = '<div class="loading">Failed to load catalog data. Make sure catalog_data.js is loaded.</div>';
    }
});
