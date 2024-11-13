import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def load_and_predict(text: str, model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval() 
    
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    sentiment = "positive" if predicted_class == 1 else "negative"
    return {"sentiment": sentiment, "confidence": confidence}


def get_sentiment(text):
    if os.path.exists(os.path.join('checkpoint', 'bert', 'final_model')):
        return load_and_predict(text,os.path.join('checkpoint', 'bert', 'final_model'))
    elif os.path.exists(os.path.join('checkpoint', 'distilbert', 'final_model')):
        return load_and_predict(text,os.path.join('checkpoint', 'distilbert', 'final_model'))
    elif os.path.exists(os.path.join('checkpoint', 'roberta', 'final_model')):
        return load_and_predict(text,os.path.join('checkpoint', 'roberta', 'final_model'))
    else:
        return "No model found. Please check the checkpoint folder."

iface = gr.Interface(
    fn=get_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to get sentiment analysis."
)

iface.launch()
