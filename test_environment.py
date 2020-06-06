import sys
REQUIRED = 'python3'

def main():
    system_major = sys.version_info.major

    if REQUIRED == 'python':
        required_major = 2
    elif REQUIRED == 'python3':
        required_major = 3
    else:
        raise ValueError(f'Unrecognized python interpreter: {REQUIRED_PYTHON}')

    if system_major != required_major:
        raise TypeError(
            f'This project requires Python {required_major}.',
            f'Found: Python {sys.version}')
    else:
        print('Development environment passes all tests!')

if __name__ == '__main__':
    main()
