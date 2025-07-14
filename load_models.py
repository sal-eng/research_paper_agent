#%% Import necessary libraries

from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv

#%% Load environment variables from .env file
load_dotenv()


#%% Load the models
GEMINI_MODEL = GeminiModel('gemini-2.5-flash-preview-04-17', provider='google-gla')
# %%
