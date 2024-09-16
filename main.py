import os 
import gradio as gr
from dotenv import load_dotenv

from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSizeConversation import MaxSizeConversation

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")

llm = GroqModel(api_key=API_KEY)

allowed_bots = llm.allowed_models


conversation=MaxSizeConversation()

agent=SimpleConversationAgent(conversation=conversation, llm=llm)

def load_model(selected_model):
    return GroqModel(api_key=API_KEY, name=selected_model)

def converse(text : str,history, system_context , model_name):    
    llm = load_model(model_name)
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    # agent.conversation.system_context = SystemMessage(content=system_context)
    input_message = str(text)
    result = agent.exec(input_message)    
    return str(result)


demo = gr.ChatInterface(
    fn=converse,
    additional_inputs=[
        gr.Textbox(label="System Context"),
        gr.Dropdown(label="Model Name", choices=allowed_bots, value= allowed_bots[0]),
    ],
    title = "Swarmauri Chatbot",
    description="Chatbot powered by Swarmauri",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",server_port=8000)