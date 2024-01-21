import logging
import os
from datetime import datetime
logs_path=os.path.join(os.getcwd(),"logs")
os.makedirs(logs_path,exist_ok=True)
day_logs_path=os.path.join(logs_path,f"{datetime.now().strftime('%d_%m_%Y')}")
os.makedirs(day_logs_path,exist_ok=True)
specific_logs_path=os.path.join(day_logs_path,f"{datetime.now().strftime('%H_%M_%S')}.log")
logging.basicConfig(
    filename=specific_logs_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
    )