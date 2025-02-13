import json
import openai
import time

openai.api_key = ""

provided_personas = [
    "Ethan", "Maya", "Elena", "Sophia", "James", "Marco",
    "Noah", "Gabriel", "Linda", "Maria", "Alex", "Priya", "Laura"
]

depression_symptoms = {
    "sad": "Sadness",
    "fatigue": "Fatigue",
    "tired": "Fatigue",
    "hopeless": "Hopelessness",
    "worthless": "Worthlessness",
    "guilty": "Guilt",
    "overwhelmed": "Overwhelmed",
    "sleep": "Sleep Disturbance",
    "insomnia": "Sleep Disturbance",
    "concentration": "Concentration Problems",
    "focus": "Concentration Problems",
    "interest": "Loss of Interest",
    "motivation": "Loss of Interest"
}


def chat_with_persona(persona_name, user_prompt, max_turns=5):
    messages = [
        {"role": "system", "content": f"You are {persona_name}, a friendly AI."},
        {"role": "user", "content": user_prompt}
    ]
    
    conversation = [{"role": "user", "message": user_prompt}]
    
    for _ in range(max_turns):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        assistant_message = response["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_message})
        conversation.append({"role": persona_name, "message": assistant_message})

        # Stop early if depressive signals are detected
        if any(word in assistant_message.lower() for word in depression_symptoms.keys()):
            break
        
        time.sleep(1)

    return {"LLM": persona_name, "conversation": conversation}
