
## Data Preparation

The model expects input features of shape (batch, 60, 28) — 60 trading days × 28 alpha factors.

### Input Format
Each stock-day sample is a flattened vector of 1680 features (60 days × 28 factors).
The 28 daily factors include:
- Price-based: open, high, low, close, returns (5 features)
- Volume-based: volume, turnover, volume ratio (3 features)
- Technical: RSI, MACD, Bollinger Bands, etc. (14 features)
- Market context: index returns, sector returns, volatility (3 features)
- Derived: cross-sectional rank percentiles (3 features)

### Data Preparation Script
```python
# Example: prepare data for inference
import numpy as np
import torch

def prepare_features(stock_data, lookback=60):
    # stock_data: dict mapping stock_code -> numpy array (T, 28)
    X = []
    for code, data in stock_data.items():
        if len(data) >= lookback:
            for t in range(lookback, len(data)):
                X.append(data[t-lookback:t])  # (60, 28)
    return torch.tensor(np.array(X), dtype=torch.float32)
```

### Data Sources
The training data was sourced from Chinese A-share market data providers (Wind, JoinQuant, or similar).
Due to licensing restrictions, we cannot redistribute the raw data.
Contact the authors for access to the processed feature cache used in our experiments.
