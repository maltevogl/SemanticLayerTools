from semanticlayertools.cleaning.text import htmlTags, lemmaSpacy


def test_htmlclean():
    """Test removal of html tags."""
    testtext = "This He<SUB>3</SUB> is really cool, super<SUP>2</SUP> cool!"
    resultString = "This He_3 is really cool, super^2 cool!"
    assert htmlTags(testtext) == resultString


def test_lemmaSpacy():
    """Test lemmatizing with Spacy."""
    testtext = "In this paper we analyze the difficulties of gravity in rotating black holes."
    resultString = "paper analyze difficulty gravity rotate black hole"
    assert lemmaSpacy(testtext) == resultString
