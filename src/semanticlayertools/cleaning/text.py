import re
import spacy

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")


def lemmaSpacy(text):
    """Clean text using Spacy english language model.

    A spacy doc is created using the text. For each token which is not a
    stopword and longer then 3 letters the lemma is returned in lowered form.
    For historical reasons, input can also be of the form
    text = list("Actual text"), which sometimes results from data harvesting.
    In these cases only the first element is considered!

    :param text: Input text
    :type text: str
    """
    try:
        if isinstance(text, list):
            text = text[0]
        doc = nlp(text)
        tokens = ' '.join(
            [t.lemma_ for t in doc if not t.is_stop and len(t) > 3]
        )
        return tokens.lower()
    except Exception:
        raise


def htmlTags(text):
    """Reformat html tags in text using replacement list..

    Some specific html formating leads to confusion with sentence and token
    border detection. This method outputs the cleaned
    text using a replacement list.

    :param text: Input text
    :type text: str
    """
    if isinstance(text, list):
        text = text[0]
    for tagPair in [
        ('<SUB>', '_'),
        ('</SUB>', ''),
        ('<SUP>', '^'),
        ('</SUP>', '')
    ]:
        text = re.sub(tagPair[0], tagPair[1], text)
    return text
