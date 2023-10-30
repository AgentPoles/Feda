
#importing dependencies
import streamlit as st
import requests
import os
import base64
import streamlit as st
import requests
import numpy as np
import flwr as fl
from flwr.common import parameter
import tensorflow as tf
from web3 import Web3
import io
import logging
import torch
from keras.preprocessing.sequence import pad_sequences
from model import create_model, train_model, predict_next_word, tokenizer, predict_top_n_words_local_global_unique_words, train_local_model
from st_keyup import st_keyup
from keras.models import load_model
import random
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from chat import query_gpt
from data_manager import load_model_from_file, save_model_to_file, load_tokenizer, save_tokenizer, save_chat_history, prepare_history_data_for_training

# Initialize the WordNet lemmatizer


logging.getLogger("client").setLevel(logging.DEBUG)


CHAT_HISTORY = "chat_history"

# Flower Client

class TextClient(fl.client.NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def update_data(self, data):
        self.data = data

    def get_parameters(self, config):
        print("getting params")
        return [np.array(w) for w in self.model.get_weights()]

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print("fitting data")
        # The text data should be passed in config
        self.model, self.X_test, self.Y_test = train_model(self.data, self.model)
        print("fitted weights", len(self.model.get_weights()))
        return [np.array(w) for w in self.model.get_weights()], len(self.data), {}

    def evaluate(self, parameters, config):
        try:
            self.model.set_weights(parameters)
            loss, accuracy = self.model.evaluate(self.X_test, self.Y_test)
            return loss, len(self.X_test), {"accuracy": accuracy}
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return (0.0, 0, {"accuracy": 0.0})



file_ = open("./img/feda.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


def load_base_models(username):
    print("loading base models")
    model = load_model_from_file('base_model.h5', "models")
    st.session_state['localmodel'] = model
    st.session_state['globalmodel'] = model
    st.session_state['usertokenizer'] = load_tokenizer("tokenizers","tokenizer.pickle")
    st.session_state['isBase'] = True
    save_user_local_model(username, model)
    save_user_global_model(username, model)


def load_user_models(username):
     print("loading user models")
     st.session_state['isBase'] = False
     local_model_name = f"{username}_local_model.h5"
     global_model_name = f"{username}_global_model.h5"
     user_tokenizer_name = f"{username}_tokenizer.pickle"
     st.session_state['localmodel'] = load_model_from_file(local_model_name, "user_local_models")
     st.session_state['globalmodel'] = load_model_from_file(global_model_name, "user_global_models")
     st.session_state['usertokenizer'] = load_tokenizer("user_tokenizers", user_tokenizer_name)
     

def load_user_or_base_models(username):
    local_model_name = f"{username}_local_model.h5"
    global_model_name = f"{username}_global_model.h5"
    local_model_path = os.path.join("user_local_models", local_model_name)  
    global_model_path = os.path.join("user_global_models", global_model_name)  
    # Construct the path with the directory.
    if os.path.exists(local_model_path) and os.path.exists(global_model_path):
         load_user_models(username)
    else: load_base_models(username)
     

def save_user_models(username, local_model, global_model, tokenizer):
     local_model_name = f"{username}_local_model.h5"
     global_model_name = f"{username}_global_model.h5"
     user_tokenizer_name = f"{username}_tokenizer.pickle"
     save_model_to_file(local_model,local_model_name,"user_local_models")
     save_model_to_file(global_model,global_model_name,"user_global_models")
     save_tokenizer(tokenizer,"user_tokenizers",user_tokenizer_name)

def save_user_local_model(username, local_model):
     local_model_name = f"{username}_local_model.h5"
     user_tokenizer_name = f"{username}_tokenizer.pickle"
     save_model_to_file(local_model,local_model_name,"user_local_models")
     save_tokenizer(tokenizer,"user_tokenizers",user_tokenizer_name)

def save_user_global_model(username, global_model):
     global_model_name = f"{username}_global_model.h5"
     user_tokenizer_name = f"{username}_tokenizer.pickle"
     save_model_to_file(global_model,global_model_name,"user_global_models")
     save_tokenizer(tokenizer,"user_tokenizers",user_tokenizer_name)

def load_none_state():
    st.session_state['localmodel'] =  None
    st.session_state['globalmodel'] = None
    st.session_state['usertokenizer'] = None
    st.session_state['isBase'] = True
     

col_a, col_b =  st.columns(2)
st.markdown("""
    <style>
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: -2rem;
    }
    </style>
    """,unsafe_allow_html=True)
# Add components to the first column
with col_a:
    icon_col, title_col = st.columns([1, 10])  # adjust the width ratio as needed

    with icon_col:
        # Style the button to remove any border or background, making it look more like an icon.
        custom_button_style = """
            <style>
                .btn-outline-secondary {
                    background-color: transparent !important;
                    border-color: transparent !important;
                }
            </style>
        """
        st.markdown(custom_button_style, unsafe_allow_html=True)

        if st.button("📮", key="icon_button", use_container_width=True):
             if st.session_state.user_name is None:
                  pass
            #  else: try_to_re_train_model(st.session_state.user_name)

    with title_col:
        st.title("FEDA")
        
with col_b:
    col5, col1, col2, col3 = st.columns([3,1,1,1])
    st.markdown("""
    <style>
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: -2rem;
    }
    </style>
    """,unsafe_allow_html=True)
    # Add components to the first column
    with col1:
        if st.button("🖥️"):
             username = st.session_state["user_name"]
             if username is not None:
                 filename = os.path.join(CHAT_HISTORY, f"{username}_chat_history.txt")
                 if os.path.exists(filename):
                      data, chat_history = prepare_history_data_for_training(username)
                      model = st.session_state['localmodel']
                      tokenizer = st.session_state['usertokenizer']
                      if model and tokenizer:
                           st.session_state['localmodel'],st.session_state['usertokenizer'],a,b = train_local_model(data,chat_history,tokenizer,username,model)
                           
                 else:
                      st.warning(body="no chat history found", icon="🚨")

        
    with col2:
        st.button("🌍")

    with col3:
        st.button("🧬")


    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """,unsafe_allow_html=True)


# Check if modal should be shown









    
is_visible = True

#card container
container_height = 200
background_color = "#F2F2F2" 
border_color = "#CCCCCC"  
border_radius = 10  
row_spacing = "20px"  
column_spacing = "20px" 

if is_visible:
        # two rows with two columns each
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # Function to create a container with custom CSS styling
        def create_container(height, bg_color, border_color, radius, margin):
            container_style = f"height: {height}px; background-color: {bg_color}; border: 2px solid {border_color}; padding: 20px; border-radius: {radius}px; margin: {margin};"
            return container_style
        
        # creating containers and adding them to the layout with spacing
        row1_col1.markdown(
            f'<div style="{create_container(container_height, background_color, border_color, border_radius, margin=row_spacing)}">' 
            f'<div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 24px; font-weight: bold; color:#000000; height: 100%;">'
            f'FEDERATED</div> </div>',
            unsafe_allow_html=True
        )
        row1_col2.markdown(
            f'<div style="{create_container(container_height, background_color, border_color, border_radius, margin=row_spacing)}">' 
            f'<div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 24px; font-weight: bold; color:#000000; height: 100%;">'
            f'AND DECENTRALIZED </div> </div>',
            unsafe_allow_html=True
        )
        row2_col1.markdown(
            f'<div style="{create_container(container_height, background_color, border_color, border_radius, margin=row_spacing)}">' 
            f'<div style="display: flex; justify-content: center; align-items: center; text-align: center; font-size: 24px; font-weight: bold; color:#000000; height: 100%;">'
            f'AUTOCOMPLETER </div> </div>',
            unsafe_allow_html=True
        )
        row2_col2.markdown(
           f'<div style="{create_container(container_height, background_color, border_color, border_radius, margin=row_spacing)}">'
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="data:image/png;base64,{data_url}"style="max-width: 160px; max-height: 160px;" alt="Image">'
            f'</div> </div>',
            unsafe_allow_html=True
        )

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




# SECTION 4C
#>>>>>>>>>>>>>>>>> CHAT DISPLAY AND LOGIC >>>>>>>>>>>>
def set_markdown_style():
    # Defining the new style for inline code (you can change the color to your preference)
    new_style = """
    <style>
        code {
            color: #FF5733;  /* Change to your preferred color */
        }
    </style>
    """

    st.markdown(new_style, unsafe_allow_html=True)  # Apply the new style

# Remember to call this function early in your app to set the style globally
set_markdown_style()




if "user_name" not in st.session_state:
    st.session_state.user_name = None
# initializing state
if 'localmodel' not in st.session_state:
    load_none_state()

if 'globalmodel' not in st.session_state:
    load_none_state()

if 'isBase' not in st.session_state:
    load_none_state()

if 'usertokenizer' not in st.session_state:
    load_none_state()

if "messages" not in st.session_state:
    st.session_state.messages = []










# Displaying chat messages from history on app rerun
for message in st.session_state.messages:
    if(message["role"] == 'assistant'):
        with st.chat_message(message["role"]):
                    response = message["content"]
                    # prepare_output(response)
                    st.markdown(response)

    else:
        with st.chat_message(message["role"],avatar="🐒"):
                    response = message["content"]
                    st.markdown(response)

content_placeholder = st.empty()
assistant_placehoder = st.empty()



if st.session_state.user_name is None:
    prompte = st.chat_input("Hello! Before we begin, what's your name?")
    if prompte:
        st.session_state.user_name = prompte  # save the name in session state
        st.session_state.messages.append({"role": "user", "content": prompte})
        
        load_user_or_base_models(prompte)
          
        with st.chat_message("user", avatar="🐒"):
            st.markdown(prompte)
        with st.chat_message("assistant"):
            response = f"Nice to meet you, {prompte}!"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.experimental_rerun()


else:
        # prompt = st.chat_input(f"Howdy, {st.session_state.user_name}! Let's chat.")
        # Create two columns for the input and button.
         # Adjust the width ratio as needed.

        # Place a text input in the left column.
            instr = 'Howdy, tell me something'
           
            st.write("")
      
            col1, col2 = st.columns([8,1]) 

            # Use the first column for text input
            with col1:
                prompt = st_keyup(
                    instr,
                    placeholder=instr,
                    debounce=500,
                    label_visibility='collapsed'
                )
            # Use the second column for the submit button
                if prompt:
                    
                    if st.session_state.get('localmodel') is None:
                        st.warning("Train the model first.")

                    else:
                        try:
                            # Get list of predicted words
                            predicted_words = predict_top_n_words_local_global_unique_words(
                                st.session_state['localmodel'],st.session_state['globalmodel'], st.session_state['usertokenizer'], prompt)

                            # Format the predicted words for markdown
                            words_markdown = "   ".join(
                                [f"`{word}`" for word in predicted_words])

                            # Display predicted words in a row
                            st.markdown(words_markdown)

                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")
            with col2:
                submitted = st.button('Chat')
            
                if prompt and submitted:
                    
                    with content_placeholder:
                        model_from_session = st.session_state['localmodel']
                        print(model_from_session.summary())
                   
                        with st.chat_message("user", avatar="🐒"):
                            st.markdown(prompt)
                            # save_chat_to_file(st.session_state.user_name, prompt)
                            save_chat_history(prompt, st.session_state['user_name'])
                            st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    
                    with assistant_placehoder:

                        with st.chat_message("assistant"):
                            # Example response for demonstration
                            response = query_gpt(prompt)
                            st.markdown(response) 
                            st.session_state.messages.append({"role": "assistant", "content": response})
                    
 