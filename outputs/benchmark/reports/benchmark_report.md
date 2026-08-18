# Benchmark Report - FuelSentinel-AI

**Date:** 2026-07-31 05:26:55
**Number of models:** 9

## Model Ranking

| Rank | Model | Score | Accuracy | F1 (Weighted) | Bal. Accuracy | Params | Size (MB) | Train Time (s) |
|------|-------|-------|----------|---------------|---------------|--------|-----------|----------------|
| 1 | cnn_gru | 100.0 | 0.8502 | 0.8433 | 0.6546 | 386628.0 | 1.47 | 1581.1396262645721 |
| 2 | gru | 99.0 | 0.8535 | 0.8360 | 0.5832 | 313092.0 | 1.19 | 1541.567170381546 |
| 3 | bigru | 92.2 | 0.8535 | 0.8466 | 0.6365 | 594692.0 | 2.27 | 1578.7249376773834 |
| 4 | tcn | 86.9 | 0.8350 | 0.8060 | 0.5234 | 619588.0 | 2.36 | 2730.196984767914 |
| 5 | gru_attention | 77.3 | 0.8468 | 0.8402 | 0.6254 | 793092.0 | 3.03 | 3453.2532086372375 |
| 6 | lstm_attention | 66.2 | 0.8401 | 0.8221 | 0.5606 | 926212.0 | 3.53 | 3456.392011642456 |
| 7 | cnn_bilstm | 52.5 | 0.8114 | 0.7696 | 0.4523 | 880708.0 | 3.36 | 1614.5854241847992 |
| 8 | bilstm | 48.0 | 0.7963 | 0.7341 | 0.3422 | 727812.0 | 2.78 | 1584.163194656372 |
| 9 | transformer | 16.8 | 0.7475 | 0.6395 | 0.2500 | 758020.0 | 2.89 | 2112.389335155487 |

## Recommendation
**Best Model:** cnn_gru

This model was selected based on the weighted score considering:
- accuracy: 20%
- f1_weighted: 20%
- balanced_accuracy: 15%
- inference_speed: 15%
- model_size_mb: 10%
- params: 10%
- training_time: 10%

## Model Details

### bilstm

| Metric | Value |
|--------|-------|
| accuracy | 0.7963 |
| f1_weighted | 0.7341 |
| balanced_accuracy | 0.3422 |
| inference_speed | 0.0237 |
| model_size_mb | 2.7800 |
| params | 727812 |
| training_time | 1584.1632 |

### gru

| Metric | Value |
|--------|-------|
| accuracy | 0.8535 |
| f1_weighted | 0.8360 |
| balanced_accuracy | 0.5832 |
| inference_speed | 0.0240 |
| model_size_mb | 1.1900 |
| params | 313092 |
| training_time | 1541.5672 |

### bigru

| Metric | Value |
|--------|-------|
| accuracy | 0.8535 |
| f1_weighted | 0.8466 |
| balanced_accuracy | 0.6365 |
| inference_speed | 0.0239 |
| model_size_mb | 2.2700 |
| params | 594692 |
| training_time | 1578.7249 |

### cnn_bilstm

| Metric | Value |
|--------|-------|
| accuracy | 0.8114 |
| f1_weighted | 0.7696 |
| balanced_accuracy | 0.4523 |
| inference_speed | 0.0241 |
| model_size_mb | 3.3600 |
| params | 880708 |
| training_time | 1614.5854 |

### cnn_gru

| Metric | Value |
|--------|-------|
| accuracy | 0.8502 |
| f1_weighted | 0.8433 |
| balanced_accuracy | 0.6546 |
| inference_speed | 0.0239 |
| model_size_mb | 1.4700 |
| params | 386628 |
| training_time | 1581.1396 |

### transformer

| Metric | Value |
|--------|-------|
| accuracy | 0.7475 |
| f1_weighted | 0.6395 |
| balanced_accuracy | 0.2500 |
| inference_speed | 0.0238 |
| model_size_mb | 2.8900 |
| params | 758020 |
| training_time | 2112.3893 |

### tcn

| Metric | Value |
|--------|-------|
| accuracy | 0.8350 |
| f1_weighted | 0.8060 |
| balanced_accuracy | 0.5234 |
| inference_speed | 0.0209 |
| model_size_mb | 2.3600 |
| params | 619588 |
| training_time | 2730.1970 |

### lstm_attention

| Metric | Value |
|--------|-------|
| accuracy | 0.8401 |
| f1_weighted | 0.8221 |
| balanced_accuracy | 0.5606 |
| inference_speed | 0.0224 |
| model_size_mb | 3.5300 |
| params | 926212 |
| training_time | 3456.3920 |

### gru_attention

| Metric | Value |
|--------|-------|
| accuracy | 0.8468 |
| f1_weighted | 0.8402 |
| balanced_accuracy | 0.6254 |
| inference_speed | 0.0225 |
| model_size_mb | 3.0300 |
| params | 793092 |
| training_time | 3453.2532 |
