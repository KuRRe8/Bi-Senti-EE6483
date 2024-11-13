import gradio as gr

def get_sentiment(text):
    # Placeholder function to process the input text
    # Replace this with your actual sentiment analysis logic
    return f"Sentiment analysis result for: {text}"

iface = gr.Interface(
    fn=get_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to get sentiment analysis."
)

def grrun():
    iface.launch()

##
grrun()