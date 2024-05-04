# here we define additional functions that will be used in the text evaluator file
# The approach proposed is to first get the input text, translate it to english and use the metrics
# for evaluating the english text.

import requests
import  os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('Yandex_api_key')
folder_id = os.getenv('folder_id')
model_uri = os.getenv('yandex_model_uri')

# Import Yandex GPT
from langchain_community.llms import YandexGPT 

class TextEvaluator:
    def __init__(self, text, api_key, model_uri):
        self.text = text
        self.gpt = YandexGPT(api_key=api_key, model_uri=model_uri)

    def detect_language(self):
        return self.gpt.detect_language(self.text)

    def translate_to_english(self):
        detected_language = self.detect_language()
        
        if detected_language == 'ru':
            translated_text = self.gpt.translate(self.text, target_language='en')
            return translated_text
        else:
            return "Text is not in Russian, cannot translate."


    def translate(self):
        '''
        This method detects the language and translates it to English.
        '''
        translated_text = self.translate_to_english()
        if translated_text:
            print("Translated text:", translated_text)
            return translated_text

    def smog_index(self):
        pass

    def coleman_liau_index(self):
        pass

    def gunning_fog(self):
        pass

    def dale_chall(self):
        pass

    def automated_readability_index(self):
        pass
