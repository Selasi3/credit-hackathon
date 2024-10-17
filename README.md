## Loan Default Prediction API

This API predicts whether a loan will **default** or **not default** based on various input features such as home ownership status, employment status, loan amount, and more. It utilizes a Gradient Boosting classifier, with pre-processing steps including label encoding for categorical features and feature scaling for numerical inputs.

### Features

- **POST** request to the `/predict` endpoint for loan default prediction.
- Encodes categorical features (e.g., `home`) and scales numerical features.
- Returns a human-readable prediction: `"Default"` or `"No Default"`.