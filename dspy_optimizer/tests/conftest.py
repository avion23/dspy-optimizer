import pytest
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="session")
def configure_dspy():
    if os.getenv("GEMINI_API_KEY"):
        lm = dspy.LM(
            model="openai/gemini-1.5-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY")
        )
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            pytest.skip("No API keys found in environment")
        lm = dspy.LM(model="openai/gpt-4o-mini")
    
    dspy.settings.configure(lm=lm)
    return lm

@pytest.fixture
def style_samples():
    return [
        {
            "name": "formal_academic",
            "sample": "The results indicate a statistically significant correlation between variables X and Y (p < 0.01). Furthermore, the analysis reveals several underlying factors that warrant further investigation.",
            "content_to_style": "This shows X and Y are related. We found some interesting things worth looking into more."
        },
        {
            "name": "casual_conversational",
            "sample": "Hey there! Just wanted to let you know that I checked out that new restaurant we talked about. It was pretty awesome, though a bit pricey. The desserts were to die for!",
            "content_to_style": "I visited the restaurant we discussed previously. It was excellent but expensive. The desserts were exceptional."
        }
    ]