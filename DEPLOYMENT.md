# Deploying the Dashboard to Streamlit Cloud

## Prerequisites
- GitHub account with the repo pushed publicly
- Streamlit Cloud account (free) at [share.streamlit.io](https://share.streamlit.io)

## Steps

### 1. Add the dataset to the repo (or use a workaround)

Streamlit Cloud can't access local files. Two options:

**Option A — Commit a sample dataset (recommended for demo)**
```bash
# Create a 5,000 row sample safe to commit
python3 -c "
import pandas as pd
df = pd.read_csv('data/games.csv').sample(5000, random_state=42)
df.to_csv('data/games_sample.csv', index=False)
"
git add data/games_sample.csv
git commit -m "feat: add sample dataset for Streamlit Cloud deployment"
git push
```
Then update `config.py` to fall back to `games_sample.csv` when `games.csv` is not present.

**Option B — Host data on Google Drive / S3**
Upload `games.csv` to a public Google Drive link and fetch it on first load using `gdown`.

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Select your GitHub repo: `harshp9945/lol-match-analytics`
4. Set **Main file path**: `dashboard/app.py`
5. Click **Deploy**

Streamlit Cloud will install `requirements.txt` automatically and give you a public URL like:
```
https://lol-match-analytics.streamlit.app
```

### 3. Update README with live link

Replace the placeholder in README.md:
```markdown
> 🚀 **[Launch Interactive Dashboard →](https://lol-match-analytics.streamlit.app)**
```

## Local Development

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Dashboard runs at `http://localhost:8501`
