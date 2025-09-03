PRINT_LEVEL = 0 # 0 : all, 1 : only warnings & exceptions, 2 : only exceptions, 3 : nothing

class Printer:
    """
        Every print should be performed through this class with appropriate level of priority (Printer.print(..., level=int))
    """
    def print(*args, level=0):
        if level >= PRINT_LEVEL:
            print(*args)