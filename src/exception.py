import sys
from src.logger import logging

def error_msg_deatils(error,error_deatils:sys):
    _,_,exc_tb=error_deatils.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_msg="error are occured in the script name [{0}] Line number [{1}] Error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_msg


class CustomExceptions(Exception):
    def __init__(self,error_msg,error_deatils:sys):
        super().__init__(error_msg)
        self.error_msg=error_msg_deatils(error_msg,error_deatils=error_deatils)
    def __str__(self):
        return self.error_msg
        