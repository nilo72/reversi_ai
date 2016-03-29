def prop_parse(props):
    """Given an input list of strings, for each string that has an = in it,
    return a mapping of the left halves to the right halves.  For a string to
    end up in the map it must have format key=value, forming {key: value}"""
    result = {}

    for arg in props:
        split = arg.split('=')
        if len(split) != 2:
            continue
        property = split[0]
        value = split[1]
        result[property] = value

    return result
