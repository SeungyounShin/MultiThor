def colorprint(text, color='black', font='original'):
    # ex) colorprint("Hello, world!", color='pink', font='italic')
    color_codes = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m',
        'orange': '\033[38;5;208m',  # Additional color option
        'pink': '\033[38;5;200m',    # Additional color option
        'gray': '\033[38;5;240m'     # Additional color option
}

    font_styles = {
        'bold': '\033[1m',
        'underline': '\033[4m',
        'italic': '\033[3m',
        'strikethrough': '\033[9m',
        'reset': '\033[0m'
    }

    colored_text = f"{color_codes.get(color, '')}{font_styles.get(font, '')}{text}{font_styles['reset']}{color_codes['reset']}"
    print(colored_text)
