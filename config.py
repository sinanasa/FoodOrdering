import os
from dotenv import load_dotenv

# load dotenv
def load_config():
    return load_dotenv()

# function to get groq api key
def get_groq_api():
    # return os.getenv('GROQ_API_KEY')
    return "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"

def get_openai_api():
    return "sk-proj-TW94wyFGqSajbRdwAquFT3BlbkFJrMPMPiGyFknoTg6puHIe"