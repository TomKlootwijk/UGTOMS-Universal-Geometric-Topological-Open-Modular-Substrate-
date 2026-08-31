def is_balanced_brackets(text):
    stack = []
    brackets = {')': '(', ']': '[', '}': '{'}
    
    for char in text:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != brackets[char]:
                return False
        else:
            continue
    
    return len(stack) == 0
