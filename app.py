import os
from openai import OpenAI
import streamlit as st

api_key = os.getenv("NVIDIA_API_KEY")

# Check if the API key is found
if api_key is None:
    st.error("NVIDIA_API_KEY environment variable not found.")
else:
    # Initialize the OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

class ConversationManager:
    def __init__(self):
        # (No need to initialize history here, it will be handled in main())
        pass

    def generate_ai_response(self, prompt):
        """Generates a response from an AI model

        Args:
        prompt: The prompt to send to the AI model.

        Returns:
        response from the AI model.
        """
        try:
            # Access conversation_history from session state
            messages = [
                {
                    "role": "system",
                    "content": "You are a programming assistant focused on providing \
                    accurate, clear, and concise answers to technical questions. \
                    Your goal is to help users solve programming problems efficiently, \
                    explain concepts clearly, and provide examples when appropriate. \
                    Use a professional yet approachable tone. Use explicit markdown \
                    format for code for all codes in the output."
                }
            ]

            for message in st.session_state.conversation_manager.conversation_history:
                messages.append(message)

            messages.append({
                "role": "user",
                "content": prompt
            })

            completion = client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                temperature=0.5,  # Adjust temperature for creativity
                top_p=1,
                max_tokens=1024,
                messages=messages,
                stream=False
            )

            model_response = completion.choices[0].message.content

            st.session_state.conversation_manager.conversation_history.append({
                "role": "assistant",
                "content": model_response
            })

            st.session_state.conversation_manager.conversation_history.append({
                "role": "assistant",
                "content": completion.choices[0].message.content
            })                
            return model_response
            
        except Exception as e:
            st.error(f"Error handling AI response: {e}")
            return None


def main():
    # Initialize ConversationManager in session state if not already present
    if "conversation_manager" not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
        st.session_state.conversation_manager.conversation_history = []

    st.title("AI-Assisted Code Generator")

    tab1, tab2, tab3 = st.tabs(["About", "Code Generation",  "Conversation History"])

    with tab1:
        st.header("About this App")
        st.write("This app demonstrates how to use AI to assist in code generation.")

    with tab2:
        st.header("Generate Code")
        framework = st.selectbox("Select a framework", ["Streamlit", "Gradio"])
        app_details = st.text_area("Describe the app you want to create", value="Create a complete app that ")

        if st.button("Generate Prompt"):
            user_prompt = f"Using {framework}, {app_details}"
            st.write("**Generated Prompt:**", user_prompt)

            with st.spinner("Thinking..."):
                # Add the user message to the history FIRST
                st.session_state.conversation_manager.conversation_history.append({"role": "user", "content": user_prompt})  

                ai_response = st.session_state.conversation_manager.generate_ai_response(user_prompt)
                if ai_response:
                    st.session_state.conversation_manager.conversation_history.append({"role": "assistant", "content": ai_response})
                    st.markdown(f"**User:** {user_prompt}")  
                    st.markdown(f"**AI:** {ai_response}")
                else:
                    st.write("**Error:** Failed to generate AI response.")

    with tab3:
        st.header("Conversation History")
        if st.session_state.conversation_manager.conversation_history:
            for msg in st.session_state.conversation_manager.conversation_history:
                if msg['role'] == 'user':
                    st.markdown(f"**User:** {msg['content']}")
                else:
                    st.markdown(f"**AI:** {msg['content']}")
        else:
            st.write("No conversation history yet.")

if __name__ == "__main__":
    main()