import os
from dotenv import load_dotenv
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(ROOT_DIR,'.env'))