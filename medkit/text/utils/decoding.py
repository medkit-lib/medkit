__all__ = ["get_ascii_from_unicode"]

import logging

from anyascii import anyascii


def get_ascii_from_unicode(text: str, keep_length: bool = True, logger=None) -> str:
    """Function returning the (closest) ascii text when possible

    Parameters
    ----------
    text:
        The unicode text to decode to ascii
    keep_length
        If True, special characters which change the length are kept in returned string
    logger
        Logger to pass if one wants to keep caller information

    Returns
    -------
    str
        The closest ascii text
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    output = anyascii(text)

    # Verify that text length is conserved
    if keep_length and len(output) != len(text):
        # if text conversion had changed its length, only change characters with same length
        output = ""
        special_chars = set()
        for c in text:
            cprim = anyascii(c)
            if len(cprim) == 1:
                output += cprim
            else:
                output += c
                special_chars.add(c)

        logger.info(
            "Some characters can't be decoded to ascii without changing length."
            " Strategy is to keep these special characters: %s \n",
            special_chars,
        )

        logger.debug("original text:\t%s\ndecoded text:\t%s\n", text, output)

    return output
