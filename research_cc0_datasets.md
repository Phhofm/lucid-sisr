# CC0 Photo Dataset Research for SR Training (v2 lucid_cc0)

## Context
The current `lucid_cc0` dataset uses PD12M (Pinterest-scraped, CC0 but likely sharpening-biased) + UHD-IQA (4K Pixabay images, CC0). Total: ~232K filtered tiles after LUCID filtering. Goal: find better source datasets that produce a higher-quality trained SR model.

---

## Dataset Analysis

### 1. Unsplash Dataset
- **License**: Unsplash License (irrevocable, nonexclusive, worldwide; free for commercial/non-commercial; no attribution required; cannot compile to build competing service). Effectively CC0-equivalent for training purposes.
- **Size**: 6.5M+ photos in Full dataset; 25K in Lite dataset
- **Resolution**: Most photos are high-resolution (typically 3000-6000px on long edge). Pro photographer quality.
- **Quality**: Excellent. Curated by professional photographers. Consistently high aesthetic and technical quality.
- **Access**:
  - **Lite (25K photos)**: Immediate download, ~1GB, commercial OK. `https://unsplash.com/data/lite/latest`
  - **Full (6.5M+ photos)**: Non-commercial only, requires request at `https://unsplash.com/data`. ~80GB raw, ~20GB compressed. Includes metadata (keywords, searches, EXIF).
  - **API**: Free API for programmatic access, rate-limited (50 req/hr demo, higher for production).
- **Used for SR?**: Not directly, but Unsplash is the source for many curated photo datasets.
- **Verdict**: ⭐⭐⭐⭐⭐ BEST OPTION. Massive scale, exceptional quality, effectively permissive license for research/training. The non-commercial restriction on the Full dataset is the main caveat — but research training likely qualifies.

### 2. Pexels
- **License**: Pexels License (formerly CC0, now custom). Free for commercial/non-commercial use, no attribution required. Cannot sell unaltered copies, cannot imply endorsement, cannot redistribute on stock platforms.
- **Size**: 3.5M+ photos (and videos)
- **Resolution**: High-quality, typically 3000-6000px
- **Quality**: Good. Curated by editorial team. Quality can be uneven — some lower-quality submissions slip through.
- **Access**: API available (free API key). No bulk dataset download — must paginate through API. Rate limits apply.
- **Used for SR?**: Not commonly used for SR research.
- **Verdict**: ⭐⭐⭐⭐ Good option. Large, high-quality, permissive license. No bulk download is the main friction point — you'd need to crawl via API. No obvious CC0 bulk dataset available.

### 3. Pixabay
- **License**: Pixabay Content License (CC0-equivalent). Free for commercial/non-commercial, no attribution required.
- **Size**: 6.1M+ images (photos, illustrations, vectors, videos). Photos subset is smaller.
- **Resolution**: Varies. Many high-res (3000+ px), but also many lower-quality submissions.
- **Quality**: Mixed. Quality control is less strict than Unsplash. Contains illustrations/vectors mixed with photos.
- **Access**: API available (free). No bulk dataset download. Must paginate via API.
- **Used for SR?**: UHD-IQA dataset is sourced from Pixabay (pre-2019). So yes, indirectly.
- **Verdict**: ⭐⭐⭐ Decent but lower average quality than Unsplash. Contains non-photo content (illustrations, vectors). Would need significant filtering.

### 4. Wikimedia Commons
- **License**: Mixed. Many images are CC0 or Public Domain, but many are CC-BY, CC-BY-SA, or fair use. Must filter to CC0/PD subset.
- **Size**: 90M+ total files (as of 2025). CC0 subset is substantial but smaller.
- **Resolution**: Very variable. Some high-res (from museum scans, satellite imagery), many low-res thumbnails.
- **Quality**: Extremely variable. Ranges from professional photography to low-quality snapshots to historical documents. Heavy bias toward illustrations, diagrams, and non-photographic content.
- **Access**: API available. Bulk download possible via `wikimedia/datasets` on GitHub (40M+ images). Can filter by license via API.
- **Used for SR?**: Not commonly used for SR.
- **Verdict**: ⭐⭐ Too much non-photo content (illustrations, diagrams, screenshots). Quality too variable. Significant filtering needed. Not ideal for SR training.

### 5. COCO
- **License**: Annotations are CC-BY-4.0. Images are under Flickr Terms of Use (varies per image — many are CC-BY). NOT CC0.
- **Size**: 330K images
- **Resolution**: Variable, many are relatively low-res (640px long edge in some splits)
- **Quality**: Variable. Many are casual snapshots.
- **Used for SR?**: Not for SR — primarily detection/segmentation.
- **Verdict**: ❌ NOT SUITABLE. Not CC0. Low average resolution. Wrong content type for SR.

### 6. LAION
- **License**: Mixed. LAION-5B URLs point to web images with varying licenses. LAION-400M similarly. No guaranteed CC0 subset at scale.
- **Subsets**: `devourthemoon/laion-publicdomain` on HuggingFace filters for archive.org URLs and CC-licensed images, but size is unclear and likely small relative to full LAION.
- **Size**: 5.85B pairs (LAION-5B), 400M (LAION-400M). CC0 subset is unknown/small.
- **Resolution**: Variable (often 224-512px CLIP-filtered)
- **Quality**: Highly variable, noisy web data.
- **Used for SR?**: LAION-Aesthetics has been used for generative model training, not SR.
- **Verdict**: ⭐ Too noisy, no large guaranteed CC0 subset, low average resolution. Not suitable.

### 7. PxHere
- **License**: CC0. Fully public domain.
- **Size**: ~1.1M photos
- **Resolution**: High-quality, typically 3000-6000px
- **Quality**: Good. Real photographs, curated.
- **Access**: API available. HuggingFace dataset exists (`nyuuzyou/pxhere`) with ~1.1M images in WebDataset format.
- **Used for SR?**: Not commonly.
- **Verdict**: ⭐⭐⭐⭐ Excellent. True CC0, large enough, high quality. Smaller than Unsplash but cleaner license. The HuggingFace dataset is very convenient.

### 8. Snappygoat
- **License**: CC0/Public Domain (aggregates from Wikimedia, Pixabay, Flickr CC0, British Library, NYPL, etc.)
- **Size**: 14M+ images
- **Resolution**: Variable (depends on source)
- **Quality**: Very variable (aggregator, not curated)
- **Access**: Web scraping only. No API or bulk download.
- **Verdict**: ⭐⭐ Too much aggregation noise. No clean API. Quality control impossible.

### 9. Flickr CC0/Public Domain
- **License**: CC0 or Public Domain for tagged images
- **Size**: Millions of CC0-tagged photos
- **Resolution**: Variable but many high-res
- **Quality**: Variable
- **Access**: Flickr API (with rate limits). Can filter by license.
- **Used for SR?**: Flickr2K is a common SR dataset (but not CC0 — it's Flickr CC-BY).
- **Verdict**: ⭐⭐⭐ Decent source but requires significant filtering. Flickr2K is already used for SR but isn't CC0.

### 10. Other Notable Sources
- **Smithsonian Open Access**: 2.8M images, CC0. But mostly historical/art, not modern photography.
- **free-images.com**: Millions of CC0/PD images. Aggregator. Quality variable.
- **KoalaAI/StockImages-CC0**: 1000 images on HuggingFace. Too small.
- **Dreamstime CC0**: 2M+ public domain photos. Good quality but no easy bulk API.

---

## Ranked Recommendations

### 🥇 1. Unsplash Full Dataset (6.5M+ photos)
**Why**: Largest curated high-quality photo dataset with permissive license. Pro photographer quality = consistent high aesthetic/technical quality. EXIF metadata included. Already proven scale for ML research.

**Caveat**: Full dataset is non-commercial license. For research training this should be fine, but verify. Lite dataset (25K) is commercial-OK but too small for LUCID filtering at scale.

**Strategy**: Request Full dataset access. Use LUCID filtering to select ~500K-1M high-quality tiles. This alone could replace both PD12M and UHD-IQA.

### 🥈 2. PxHere Dataset (1.1M photos)
**Why**: True CC0 license (no restrictions), high-quality real photographs, convenient HuggingFace download. Smaller but cleanest license of all options.

**Strategy**: Download from HuggingFace. Combine with Unsplash for diversity. ~1M images gives plenty of room for LUCID filtering.

### 🥉 3. Pexels (3.5M+ photos, via API)
**Why**: Large, high quality, permissive license (effectively CC0 for training). Strong editorial curation.

**Caveat**: No bulk download. Must crawl via API. Rate limits. Tedious but feasible with a script.

**Strategy**: Write a crawler using Pexels API. Filter for high-resolution only (>3000px). Supplement Unsplash/PxHere with Pexels for diversity.

### 4. Pixabay (6.1M+, via API)
**Why**: Large, CC0-equivalent, API available. UHD-IQA already uses Pixabay as source.

**Caveat**: Contains illustrations/vectors. Lower average quality than Unsplash. Would need filtering to exclude non-photos.

**Strategy**: Use as supplementary source. Filter heavily for photos only, high resolution. Good for volume but not primary quality source.

### 5. Wikimedia Commons CC0 subset
**Why**: Massive scale (90M+ files), true CC0 for tagged images.

**Caveat**: Too much non-photo content. Requires aggressive filtering for real photographs. Quality too variable.

**Strategy**: Only useful as a supplementary source after aggressive filtering. Low priority.

---

## Recommended v2 Strategy

**Primary sources** (in order of priority):
1. **Unsplash Full** (~6.5M photos) — bulk of the dataset
2. **PxHere** (~1.1M photos) — clean CC0 supplement
3. **Pexels** (~3.5M photos, API crawl) — diversity supplement

**After LUCID filtering**, this should yield significantly more high-quality tiles than v1 because:
- Unsplash photos are pro-grade (vs PD12M's Pinterest-scraped casual photos)
- No social-media sharpening bias (Unsplash photos are camera-original)
- Higher average resolution means more tiles per image
- True CC0/permissive licenses throughout

**Expected yield**: With LUCID filtering at similar thresholds, ~500K-1M+ high-quality tiles from ~10M source images.
