# -*- coding: utf-8 -*-#

# ---------------------------------------------
# Name:         BackTranslationDataEnhancementMethod
# Description:  
# Author:       Laity
# Date:         2021/9/24
# ---------------------------------------------
from google_trans_new import google_translator

translator = google_translator()
ch = '你好'
en = translator.translate(ch)
print(en)