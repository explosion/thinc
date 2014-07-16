class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_namespace(field_map):
    f = Namespace()
    for field_name, value in sorted(field_map.items()):
        if type(value) == dict:
            value = make_namespace(value)
        setattr(f, field_name, value)
    return f


class ContextFields(object):
    '''
    Subclass this and write in properties.
    You'll then write your features referring to them. e.g.

    class Features(ContextFields):
        field1 = int
        field2 = int
        field3 = int

    >>> field_names = Features.field_names()
    >>> print field_names
    {'field1': 0, 'field2': 1, 'field3': 2}

    What we want to do is write templates that combine the fields in various
    ways --- and we want to write them as concisely as possible.
    

    And now we can write our templates:
    
    def make_templates(field1=0, field2=0, field3=0):
        atoms = (
          (field1,),
          (field2,),
          (field3,)
        )
    
        set1 = (
          (field1, field2)
        )

        set2 = (
          (field2, field3)
        return atoms + set1 + set2

    templates = make_templates(**field_names)

    The templates are given to Extractor, to construct masks over the context,
    that pick out the values we want:

    for i in range(template.n):
        template.values[i] = context[template.fields[i]]

    We then hash template.values to construct our feature. This part is done
    generically in Cython --- the point here is to construct a flat context
    vector, but also have a convenient way to name our features when we write
    templates. Having the templates be simple tuples is really easy for the
    writing, and we only need to "compile" them into templates once, and
    entirely generically --- so long as the tuples are just indices into
    the context.

    This really gets good when we have a nested context definition:

    class Token(ContextFields):
      word = int
      pos = int

    class ParseState(ContextFields):
      stack1 = Token
      stack2 = Token
      stack3 = Token

      depth = int
      ...

    We then get:

    >>> F = ParseState.field_names()
    >>> print F.stack1.word
    0
    >>> print F.stack2.word
    2
    >>> print F.stack3.pos
    5

    If the context has a typed field like Token, you'll usually want to have
    a helper function to set the appropriate parts of your context vector for
    you. That part's on you, though --- all we're providing here is a nested
    namespace that produces a mapping to integers.
    '''
    def field_names(cls, offset=0, ):
        namespace = Field()
        i = offset
        for field_name, value_type in sorted(cls.__dict__.items()):
            if field_name.startswith('_') or type(value_type) != type:
                continue
            elif hasattr(value_type, 'field_names'):
                value, i = value_type.field_names(offset=i)
            else:
                value = i
                i += 1
            setattr(namespace, field_name, value)
        return namespace, i
