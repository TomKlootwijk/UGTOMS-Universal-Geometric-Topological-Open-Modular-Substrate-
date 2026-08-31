def first_unique_index(text):
    char_count = {}
    for char in text:
        char_count[char] = char_count.get(char, 0) + 1
    
    for i, char in enumerate(text):
        if char_count[char] == 1:
            return i
    
    return -1
