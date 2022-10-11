import re
import spacy
import time

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print('Consider loading a larger language model. Falling back to using small english.')
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


def tokenize(
    text, languageModel=nlp, ngramRange=(1, 5), limitPOS=False,
    excludeStopWords=False, excludePunctuation=False, excludeNumerical=False,
    excludeNonAlphabetic=False,
    tokenMinLength=1, debug=False
):
    """Tokenize the provided text using the specified Spacy language model.
   
    Limit tokens to specific Parts-of-Speech by providing a list,e.g.
    limitPOS=["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"].
    Found ngrams are joined with the special character "#" (hash) which needs to be taken 
    into account in later steps of processing pipelines.
  
    Exclude stop words by setting excludeStopWords=True.
    :param languageModel: The Spacy language model used for tokenizing.
    :type languageModel: class:`spacy.nlp`
    :param ngramRange: Range of ngrams to be returned, default 1- to 5-gram.
    :type ngramRange: tuple
    :param limitPOS: Limit returned tokens to specific Parts of Speech.
    :type limitPOS: bool list
    :param excludeStopWords: Exclude stop words from returned tokens.
    :type excludeStopWords: bool
    :param tokenMinLength: Set minimal length of returned token.
    :type tokenMinLength: int
    """
    starttime = time.time()
    doc = nlp(text)
    sentList = []
    ngramList = []
    for sent in list(doc.sents):
        sentPOS = []
        for token in sent:
            if isinstance(limitPOS, list):
                if token.tag_ not in limitPOS:
                    continue
            if excludeStopWords is True:
                if token.is_stop:
                    continue
            if excludeNumerical is True:
                if token.is_digit is True:
                    continue
            if excludeNonAlphabetic is True:
                if token.is_alpha is not True:
                    continue
            if excludePunctuation is True:
                if token.is_punct is True:
                    continue
            if len(token.lemma_) >= tokenMinLength:
                sentPOS.append(token.lemma_.lower())
        sentList.append(sentPOS)
    for elem in sentList:
        for ngramLen in range(ngramRange[0], ngramRange[1] + 1, 1):
            ngrams = zip(*[elem[i:] for i in range(ngramLen)])
            ngramList.append(list(ngrams))
    if debug is True:
        print(f"Created ngrams in {time.time() - starttime} sec.")
    extractedNgrams = [x for y in ngramList for x in y]
    reformated = ['#'.join(elem) for elem in extractedNgrams]
    return reformated