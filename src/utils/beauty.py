"""
    Decorating functions are placed here
    author: spdkh
    date: Nov 2023
"""
import os


def pretty(*objects, sep=' ', end='\n', info=None, color="\033[93m"):
    reset = "\033[0m"  # Reset text color to default

    try:
        terminal_size = os.get_terminal_size().columns
    except Exception:
        terminal_size = 75
    
    block_len = min(terminal_size,
                    20 + max([len(i) for i in str(objects).split('\\n')]))

    header = 'Info:'
    print('-'*block_len)
    if info is not None:
         header += str(info)

    print(header.center(block_len))

    print('-'*block_len)

    print(color)

    print(*objects)

    print(reset)
    print('-'*block_len)


def pretty_args(args):
    text = ''
    index = 0
    for key, value in vars(args).items():
        if value is not None:
            index += 1
            text += "{:<20} = {:<10}".format(str(key), str(value))
            text += " " * 5 + '|' + " " * 5

            if index % 3 == 0:
                text += "\n"

    return text.replace(' ', '.')