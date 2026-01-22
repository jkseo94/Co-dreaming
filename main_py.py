import streamlit as st
from openai import OpenAI
from supabase import create_client
from datetime import datetime
import random
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

SYSTEM_PROMPT = """
Role: You are an AI agent designed to help users generate simulations or a subset of future-oriented thinking that involves imaginatively placing oneself in a hypothetical scenario. Your ultimate purpose is to help users mentally pre-experience the future and decision-making in intertemporal choices, such as saving.
Constraints:
- Make sure each conversation thread is less than 50 words.
- Please follow the following stages strictly. I have listed the instructions in order for you. 
- Tone: Friendly, realistic

Dialogue Steps:
Follow this sequence strictly. Do not skip steps.
1. Stage 1 — Introduction (don't show this): 
- Introduce yourself briefly as follows: "Hello! I’d like to invite you to a short session designed to think about retirement. It can sometimes feel very distant, but exploring it now helps clarify what matters to you. 

Are you ready?" 

2. Stage 2 — Small Talk (don't show this): 
- Smoothly transition from Turn 1 to Turn 2 by letting users know that you are trying to get to know them better.
- THEN ask all of the following questions one by one to get to know the user better: "How old are you right now?," "How do you describe your gender?," "How many family members do you currently have?" 

3. Stage 3 — Simulation (don't show this): 
- Guide users to mentally pre-experience what life would be like after years of saving. 
- You should help users think about what life would be like, having retired in a financially prepared state through years of saving. THEN ask them to think about a specific future event that could occur then, using the phrase “I am” as if they are there right now (e.g., “I am at the beach swimming” or “I am making coffee”).

4. Stage 4 — Pre-experience (don't show this): 
- Please help users simulate more vividly and in detailed and specific ways. 
- Ask them where they are, who they are with, and what they will be hearing and seeing at that future event they are asked to think about in Turn 3, one by one.
- Please further expand this stage 4 to actively facilitate users mentally pre-experiencing the future event.
- Break down questions into separate turns. Do not ask everything at once.
- Ensure the conversation lasts for a minimum of 5 turns and a maximum of 7 turns.

5. Stage 5 — Call to Action (don't show this): 
- Recap and synthesize the future event that the user has constructed with you during the conversation. Then ask the user how they feel about it.
- End on a hopeful note that it is not always easy to think so far ahead, but doing so is a great step toward better financial preparedness. 
- Suggest to users that saving now can help them reach a financially prepared retirement in the future, such as "Your tomorrow is built on what you do today. Why not invest in a brighter future by saving a small amount for your retirement now?"
- Please send the final message, "I hope this short conversation provided you with a meaningful perspective on your retirement."

Concluding Remarks: 
Once the users want to end the conversation after going through all five turns, provide them with a randomized finish code to proceed with the survey questionnaire.
This randomized finish code should be different for all users since it will be used to match the user's survey question answers.
Here are some issues to avoid in the conversation with the users:
1. Do not give the finish code if the users did not finish the entire conversation. If they forget to ask for the code at the end of the conversation, remember to actively offer it.
2. Ensure the user has engaged with the simulation.
"""

# ==========================================
# SERVICES (Model & Database)
# ==========================================

class DatabaseService:
    def __init__(self):
        try:
            self.supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_SERVICE_KEY"]
            )
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            self.supabase = None

    def save_full_conversation(self, finish_code, messages):
        """
        Saves the entire conversation history in one go.
        """
        if not self.supabase:
            return

        data = {
            "finish_code": finish_code,
            "full_conversation": messages,
            "finished_at": datetime.utcnow().isoformat()
        }

        try:
            self.supabase.table("full_conversations").insert(data).execute()
        except Exception as e:
            st.error(f"Failed to save conversation: {e}")

class AIService:
    def __init__(self):
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    def generate_response(self, messages, current_step):
        # Filter messages for API to remove internal state keys if any
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"You are currently responding in STEP {current_step}. Respond ONLY for this step."}
        ]

        # Add conversation history
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            response = self.client.chat.completions.create(
                model="gpt-4", # Ensure correct model name
                messages=api_messages
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI Error: {e}")
            return "Connection to 2060 interrupted."

# ==========================================
# APP LOGIC (Controller)
# ==========================================

class SimulationApp:
    def __init__(self):
        self.db = DatabaseService()
        self.ai = AIService()
        self.initialize_session_state()

    def initialize_session_state(self):
        defaults = {
            "messages": [],
            "current_step": 0, # 0=Intro, 1=Start, 2-5=Sim
            "finish_code": str(random.randint(10000, 99999)), # Generate ONCE
            "simulation_complete": False,
            "data_saved": False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def run(self):
        self.render_ui()
        self.handle_initial_message()
        self.render_chat_history()
        self.handle_user_input()

    def render_ui(self):
        st.set_page_config(page_title="Saving for the future", layout="centered")
        st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """, unsafe_allow_html=True)
        st.title("A window into the future")

    def handle_initial_message(self):
        if not st.session_state.messages:
            welcome_msg = ""
            Hello! I would like to invite you to a short session designed to think about retirement. It can sometimes feel very distant, but exploring it now helps clarify what matters to you. 

Are you ready?"""
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    def render_chat_history(self):
        for msg in st.session_state.messages:
            avatar = "🤖" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    def determine_next_step(self, assistant_text):
        """Parsing logic to determine state transitions based on AI response content"""
        text = assistant_text.lower()
        current = st.session_state.current_step

        if current == 1:
            return 2
        elif current == 2:
            return 3
        elif current == 3:
            return 4
        elif current == 4:
            return 5
        elif current == 5:
            # Stage 5 is the call to action; afterward we are done
            return 6

        return current

    def handle_user_input(self):
        if st.session_state.simulation_complete:
            st.info(f"Simulation ended. Your code: {st.session_state.finish_code}")
            return

        if prompt := st.chat_input("Type your message here"):
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Update Logic State (Start of conversation trigger)
            if st.session_state.current_step == 0:
                 if any(w in prompt.lower() for w in ["yes", "ready", "sure", "ok", "start"]):
                    st.session_state.current_step = 1

            # 3. Generate Assistant Response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Connecting to 2060..."):
                    response_text = self.ai.generate_response(
                        st.session_state.messages,
                        st.session_state.current_step
                    )

                # Logic: Check if we are at the end
                next_step = self.determine_next_step(response_text)

                # If we just finished the final step (Call to Action), append the code
                if st.session_state.current_step == 5:
                    response_text += f"\n\nYour finish code is **{st.session_state.finish_code}**."
                    st.session_state.simulation_complete = True

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # Update step for next turn
                st.session_state.current_step = next_step

            # 4. Save to DB if complete (Goal 2 & 1)
            if st.session_state.simulation_complete and not st.session_state.data_saved:
                self.db.save_full_conversation(
                    st.session_state.finish_code,
                    st.session_state.messages
                )
                st.session_state.data_saved = True

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    app = SimulationApp()
    app.run()
