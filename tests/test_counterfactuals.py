from src.counterfactuals import CounterfactualGenerator, CF2_PREFIX, CF3_SUFFIX

def test_cf1():
    assert CounterfactualGenerator.cf1('Hello, WORLD!') == 'hello world'

def test_cf2_cf3():
    assert CounterfactualGenerator.cf2('x').startswith(CF2_PREFIX + ' ')
    assert CounterfactualGenerator.cf3('x').endswith(CF3_SUFFIX)

def test_cf4_repetition():
    g=CounterfactualGenerator(['gravity pulls objects', 'objects have gravity'])
    x=g.cf4('gravity objects')
    assert x.startswith('gravity objects ')
