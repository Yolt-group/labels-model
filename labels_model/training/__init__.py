import os
import sys

# append path to be able to use imports directly from src such as `from data import xxx`
# -- since we execute train() with if __name__ == "__main__", we need to use absolute imports instead of relative .
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
