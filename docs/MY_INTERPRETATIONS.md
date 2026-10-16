# My interpretations (DRAFT - rewrite in your own words, then run: python -m src.report)

## Regression error
Row 208 had the largest test error: actual 17.3608 t, predicted 14.2267 t (error -3.1341 t, 3.9 times the generator's noise standard deviation of 0.80 t). Inputs: rainfall_mm=1308.17, avg_temp_c=18.00, soil_moisture_pct=51.31, fertilizer_kg=71.77, field_size_ha=3.23, season=A, cooperative_support=1. The generator adds random noise the model cannot observe. The selected linear model fits temperature as a straight line, while the generator penalises distance from 18 C (an absolute value); this row's temperature is 18.00 C, so this mismatch may have contributed (not tested).

## Classification error
Row 54: actual class 0, predicted 1 with probability 0.8349. Its yield per hectare is 6.066 t/ha against a dataset median of 5.640 t/ha, so it lies 0.427 t/ha from the class boundary. Total misclassifications on the test set: 5 of 65.

## Cluster
Cluster 0 is the largest (101 training farms, 53%): mostly season A. Mean yield 12.54 t, high-risk rate 46%, mean rainfall 1031 mm, mean fertilizer 126 kg, mean field size 2.06 ha. Operationally, such a group could be visited or advised together, with human review.
