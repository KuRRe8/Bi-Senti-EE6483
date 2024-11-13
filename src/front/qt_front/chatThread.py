from PySide2.QtCore import QObject, QThread, Signal, Slot
import os, sys, time
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
class ChatThread(QObject):
    updateRecordSignal = Signal(int,int,str,str)
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.model_type = None
        # lazy load model
    
    @Slot()
    def _mySendMessageSlot(self, msg):
        modified_msg = "I received: " + msg
        time.sleep(1)
        print(modified_msg)
        self.updateRecordSignal.emit(0,0,"",modified_msg)
    
    @Slot()
    def mySendMessageSlot(self, msg):
        def load_and_predict(instance, text: str, model_path: str):
            if instance.tokenizer is None:
                instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if instance.model is None:
                instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            instance.model = instance.model.to(device)
            instance.model.eval() 
            
            inputs = instance.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = instance.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            sentiment = "positive" if predicted_class == 1 else "negative"
            return {"sentiment": sentiment, "confidence": confidence}
        
        def cat_path(model_name: str):
            return os.path.abspath(os.path.join('..','..','..','checkpoint', model_name, 'final_model'))


        for model in ["bert", "distilbert", "roberta", "xlnet"]:
            model_path = cat_path(model)
            if os.path.exists(model_path):
                response=str(load_and_predict(self,msg,model_path))
                break
            nomodel = os.path.abspath(os.path.join('..','..','..','checkpoint', "model_name", 'final_model'))
            response = f"No model found in {nomodel}"


        self.updateRecordSignal.emit(0,0,"",response)