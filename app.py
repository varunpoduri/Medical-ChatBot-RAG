import asyncio
import traceback
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from graph import app


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def main():
    st.title("MEDICAL RAG Application") 

st.title("MEDICAL RAG Application")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    # Display user message
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    bot_reply = "I am unable to process your request."
    error_trace = None

    # Processing response
    with st.status("🤖 Processing... Please wait!") as status:
        try:
            response = app.invoke({
                "query": user_query,
                "chat_history": [msg for msg in st.session_state.chat_history]
            })
            
            print(f"🔍 [DEBUG] Response from app.invoke(): {response}")     
            bot_reply = response.get("generation", bot_reply)

        except Exception as e:
            error_trace = traceback.format_exc()  # Get full traceback
            bot_reply = f"⚠️ Error: {str(e)}"
            print(f"❌ [ERROR] Exception occurred: {str(e)}")
            print(f"📜 [TRACEBACK]\n{error_trace}")  

        status.update(label="✅ Response Ready!", state="complete", expanded=False)

    # Display AI response
    with st.chat_message("AI"):
        st.markdown(bot_reply)

    # Append AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=bot_reply))

    # Show error traceback if an error occurred
    if error_trace:
        with st.expander("See error details", expanded=True):
            st.code(error_trace, language="python")
