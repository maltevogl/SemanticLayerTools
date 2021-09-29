import re
import spacy

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")


def lemmaSpacy(text):
    """Clean text in dataframe column."""
    try:
        if isinstance(text, list):
            text = text[0]
        doc = nlp(text)
        tokens = ' '.join(
            [t.lemma_ for t in doc if not t.is_stop and len(t) > 3]
        )
        return tokens.lower()
    except:
        raise


def htmlTags(text):
    """Remove html tags in text."""
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
