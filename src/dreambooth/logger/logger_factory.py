import logging


class LoggerFactory:
    _configured = False

    @staticmethod
    def configure(level=logging.INFO, fmt="%(levelname)s | %(message)s"):
        """
        Configure the global logging system only once.
        """
        if not LoggerFactory._configured:
            logging.basicConfig(level=level, format=fmt)
            LoggerFactory._configured = True

    @staticmethod
    def get_logger(name: str = "DreamBooth") -> logging.Logger:
        """
        Get a logger with a consistent name.
        """
        return logging.getLogger(name)
