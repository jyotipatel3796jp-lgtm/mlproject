import sys
import logging

def error_message_detail(error, error_detail: sys):
    error_type, error_obj, error_tb = error_detail.exc_info()
    filename = error_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
filename, error_tb.tb_lineno, str(error)
    )
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    if __name__ == "__main__":
        try:
            a = 1 / 0
        except Exception as e:
            logging.info("Logging has started.")
            logging.warning("This is a warning message.")
            logging.error("This is an error message.")
            logging.info("Logging has started.")
            raise CustomException(e, sys)
