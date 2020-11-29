import logging

LOGGER = logging.getLogger(__name__)


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s', level=logging.DEBUG)
