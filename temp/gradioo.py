import gradio as gr

"""
Gradio Predefined Widgets Documentation

Available Widgets:
-----------------
1. Text Input/Output
    - Textbox: For text input/output with customizable size
    - Number: For numeric input with optional range limits
    - Slider: Numeric input with a sliding interface
    - Radio: Single-selection from predefined options
    - Checkbox: Boolean input (True/False)
    - Dropdown: Dropdown menu for option selection
    - Label: For displaying text output

2. Media Input/Output
    - Image: Handle image upload/display (supports various formats)
    - Video: Video file upload/playback
    - Audio: Audio file upload/playback
    - File: General file upload/download
    - Webcam: Capture from connected camera
    - Microphone: Record from connected mic

3. Structured Data
    - Dataframe: Display/edit tabular data
    - JSON: Handle JSON format data
    - DataFrame: For pandas DataFrame input/output
    - Plot: For matplotlib/plotly figures
    - HighlightedText: Text with customizable highlighting

4. Special Components
    - Button: Clickable interface element
    - ColorPicker: Select colors visually
    - Markdown: Format text using markdown
    - HTML: Display raw HTML content
    - Gallery: Display multiple images in grid
    - Chatbot: Two-column chat interface
    - Model3D: Display 3D models
    - TimeSeries: Time-series data visualization

Key Features:
------------
1. Input Widgets:
    - value: Set default value
    - label: Custom widget label
    - info: Additional help text
    - interactive: Enable/disable user interaction
    - visible: Control widget visibility

2. Output Widgets:
    - type: Specify output format
    - label: Custom output label
    - show_label: Toggle label visibility
    - container: Customize container properties

Common Methods:
-------------
- update(): Update widget properties
- clear(): Reset to default state
- style(): Customize appearance
- preprocess(): Transform input before processing
- postprocess(): Transform output after processing

Usage Examples:
-------------
1. Basic Text Input:
    gr.Textbox(label="Enter text", placeholder="Type here...")

2. Image Upload:
    gr.Image(label="Upload Image", type="filepath")

3. Slider with Range:
    gr.Slider(minimum=0, maximum=100, step=1, label="Select Value")

4. Dropdown Selection:
    gr.Dropdown(choices=["Option 1", "Option 2"], label="Choose")

Notes:
-----
- All widgets support both input and output modes
- Customizable styling via CSS classes
- Event listeners can be attached to any widget
- Supports batch processing and queue management
- Compatible with async/await operations
"""
# Example implementations of various Gradio widgets

def image_classifier(img):
    return "Sample Classification Result"

def text_processor(text, option, number):
    return f"hahahahahha Processed: {text} with {option} at level {number}"

def audio_transcriber(audio):
    return "Sample Transcription"

# Create a Gradio interface with multiple components
with gr.Blocks() as demo:
    gr.Markdown("# Demo of Gradio Widgets")
    
    with gr.Tab("Basic Inputs"):
        text_input = gr.Textbox(label="Enter Text", placeholder="Type here...")
        num_input = gr.Number(label="Enter Number", value=5)
        slider = gr.Slider(0, 100, label="Slide to Select")
        checkbox = gr.Checkbox(label="Check this")
        radio = gr.Radio(["Option 1", "Option 2", "Option 3"], label="Choose One")
        dropdown = gr.Dropdown(choices=["A", "B", "C"], label="Select Option")
    
    with gr.Tab("Media Inputs"):
        image_input = gr.Image(label="Upload Image")
        audio_input = gr.Audio(label="Upload Audio")
        video_input = gr.Video(label="Upload Video")
        file_input = gr.File(label="Upload File")
    
    with gr.Tab("Advanced Components"):
        df = gr.Dataframe(headers=["A", "B"], label="Edit Data")
        json_comp = gr.JSON(label="JSON Input")
        color = gr.ColorPicker(label="Pick Color")
        chat = gr.Chatbot(label="Chat Interface")

    output = gr.Textbox(label="Output")
    
    submit_btn = gr.Button("Process")
    submit_btn.click(
        text_processor,
        inputs=[text_input, dropdown, num_input],
        outputs=output
    )

demo.launch(share=True)