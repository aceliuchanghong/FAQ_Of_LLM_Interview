import os
import getpass

PROMPTLAYER_API_KEY = getpass.getpass()
os.environ["LANGCHAIN_API_KEY"] = PROMPTLAYER_API_KEY

OPENAI_API_KEY = getpass.getpass()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
