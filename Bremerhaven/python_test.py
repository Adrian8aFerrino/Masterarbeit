def wtf(strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    if len(strs) == 1:
        return strs[0]

    result = strs[0]

    # Compare characters from the second string onwards
    for string in strs[1:]:
        common_prefix = ""
        for char1, char2 in zip(result, string):
            if char1 == char2:
                common_prefix += char1
            else:
                break
        result = common_prefix
    return result


print("First test", wtf([""]))
print("Second test", wtf(["a"]))
print("Third test", wtf(["floor", "flies", "flowers"]))
print("Fourth test", wtf(["apple", "app", "apricot"]))
