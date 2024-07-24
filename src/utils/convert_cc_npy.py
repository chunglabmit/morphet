import sys
from utils.util import convert_cc_npy_ratio

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("\nUsage: python convert_cc_npy.py [CC_NPY_FILE] [Z_RATIO] [Y_RATIO] [X_RATIO]")
        sys.exit(1)
    print("converting...")
    f = sys.argv[1]
    zrat = float(sys.argv[2])
    yrat = float(sys.argv[3])
    xrat = float(sys.argv[4])

    convert_cc_npy_ratio(f, (zrat, yrat, xrat))

