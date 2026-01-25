import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    print("HF_TOKEN set:", os.environ.get("HF_TOKEN") is not None)

    from ups_challenge.dataloaders import base

    names = [x for x in dir(base) if not x.startswith("_")]
    print("\nThings available in ups_challenge.dataloaders.base:")
    for n in names:
        print(" -", n)

if __name__ == "__main__":
    main()
