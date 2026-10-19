# AI-Powered Scam and Spam Message Detection System

Simple project using:
- TF-IDF + Logistic Regression to classify messages as spam or ham.
- Gemini to explain possible warning signs and suggest safe actions.
- Streamlit for the web interface.

## Files

- `app.py`: Streamlit web application.
- `train_model.py`: trains the model and creates the `.pkl` files.
- `requirements.txt`: Python libraries needed.
- `SpamSmsDataset.xlsx`: place your dataset here before training.

## Step 1: Create the model files

Place `SpamSmsDataset.xlsx` in this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
```

This creates:
- `model.pkl`
- `vectorizer.pkl`
- `label_encoder.pkl`

Upload those three files to the same GitHub repository as `app.py`.

## Step 2: Deploy to Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and all three `.pkl` files to GitHub.
2. Deploy the repository on Streamlit Community Cloud.
3. Open your app's Settings / Secrets and add:

```toml
GEMINI_API_KEY = "paste-your-real-key-here"
```

Keep the real API key in Streamlit Secrets only. Do not put it in `app.py` or upload it to GitHub.

## Notes

- The training dataset has spam/ham labels. The model does not have a separate trained scam class.
- Gemini provides an additional language-based assessment and may make mistakes.
- The classifier was trained on SMS messages, so results on emails and chats are not validated.
- Model confidence is not a guarantee that a message is safe.
