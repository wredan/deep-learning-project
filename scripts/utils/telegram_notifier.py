import requests
import json
import pyautogui
from io import BytesIO
import os
from pytorch_lightning.loggers import TensorBoardLogger

class TelegramNotifier():

    def __init__(self) -> None:
        pass

    def sendMessage(self, logger: TensorBoardLogger):
        json_content = json.load(open(os.path.join(os.getcwd(), "bot_data.json")))
        image_byte = BytesIO()
        pyautogui.screenshot().save(image_byte, 'JPEG')
        image_byte.seek(0)
        requests.post(
                url='https://api.telegram.org/bot{0}/sendPhoto'.format(json_content['bot_token']),
                data={
                    'chat_id': json_content['chat_id'], 
                    'caption':"*{0}/{1}* ha terminato il training!".format(logger._name, logger._version),
                    'parse_mode': 'Markdown'
                    },
                files= {'photo': image_byte }
            ).json()