import logging 
import os
from datetime import datetime

Log_file=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",Log_file)
os.makedirs(os.path.dirname(logs_path),exist_ok=True)
log_file_path=os.path.join(os.getcwd(),"logs",Log_file)
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)
