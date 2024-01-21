import sys
#sys.exc_info() --->contains all the informatioin about exception currently being handled
class customException(Exception):
    def __init__(self,message,error_detail:sys):
        super().__init__(message)
        _,_,exc_tb=error_detail.exc_info()
        filename=exc_tb.tb_frame.f_code.co_filename
        self.message="\n\n\n Error Information:\nError occurred in script name {0} \nline number {1}\n error message {2}\n\n\n".format(
            filename,exc_tb.tb_lineno,message
            )
    def __str__(self):
        return self.message