"""const.py: classes for constant variables"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "07/20/2018"


class Phase(object):
    """Model Learning Phase"""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_VAL = "train_val" # for multiple phases, join by '_'
    REALTIME = "realtime"

class Dataset(object):
    """Data Set Name"""
    MNIST = "mnist"
    MICROGLIA = "microglia"
    MICROGLIA_LABELED = "microglia_labeled"
    MULTIPLEXING = "multiplexing"
    TRAP = "trap"
    REALTIME = "realtime"

class RenderingMethod(object):
    """Vispy Rendering Method for Volume"""
    MIP = "mip"
    TRANSLUCENT = "translucent"
    ADDITIVE = "additive"

class VizMarkers(object):
    """Marker names"""
    MAIN = "MainMarker"
    GFP_MICROGLIA = "GFP-Microglia"
    GFP_MICROGLIA_GOOD = GFP_MICROGLIA + "(Good)"
    GFP_MICROGLIA_BAD = GFP_MICROGLIA + "(Bad)"
    TOPRO3_NUCLEI = "ToPro3-Nuclei"
    TOPRO3_NUCLEI_NEARBY = TOPRO3_NUCLEI +"-Nearby"
    M_RAMIFIED = "Ramified"
    M_AMOEBOID = "Amoeboid"
    M_GARBAGE = "Garbage"
    ANTI_GFP_594 = "Anti-GFP-594-AxonTracing"
    VIVO = "vivo"

class NormalizationType(object):
    """Normalization method types"""
    ZERO_AND_ONE = "zero_and_one"
    MONE_AND_ONE = "minus_one_and_one"
    ZERO_MEAN = "zero_mean" # ranges vary


class StainChannel(object):
    GFP = "GFP"
    ANTIGFP = "AntiGFP"
    EGFP_ANTIGFP = "EGFP-AntiGFP"
    GFP_SEG = "GFP_SEG"
    MBP = "MBP"
    AUTOFLUORESCENCE = "AF"
    TOPRO3 = "ToPro3"
    DAPI = "DAPI"
    CY5 = "Cy5"
    CB = "CB"
    PV = "PV"
    IBA1 = "IBA1"
    IBA1_RAW = "IBA1_RAW"
    IBA1_SEG = "IBA1_SEG"
    LECTIN = "Lectin"
    UNKNOWN = "Unknown"
    UNKNOWN_SEG = "Unknown_SEG"
    UNKNOWN_SEG_GB = "Unknown_SEG_GB"
    UNKNOWN_SEG_GB_THR = "Unknown_SEG_GB_THR"
    V5 = "V5"
    V5_SEG = "V5_SEG"
    V5_SEG_GB = "V5_SEG_GB"
    V5_SEG_GB_THR = "V5_SEG_GB_THR"
    CFOS = "cFos"
    TDTOMATO = "tdTomato"


class DyeWaveLen(object):
    A546 = "Alexa546"
    UK635 = "Unknown635"
    UK559 = "Unknown559"
    UK488 = "Unknown488"
    UK647 = "Unknown647"
    UK546 = "Unknown546"


class ModelType(object):
    BMTR = "BMTR"       # default model - associated with Trainer()
    BFMC = "BFMC"       # second model with ResNet autoencoder - associated with BFMCTrainer()
    MPTR = "MPTR"       # multiplexing model
    TRAP = "TRAP"       # TRAP cFos model
    ALTR = "ALTR"       # Active Learning training model
    KMNS = "KMNS"       # K-means model for sub-clustering


class LearningMethod(object):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"


class EvalCol(object):
    # Data Frame Columns for Evaluator logs
    GID = "gid"
    FID = "fid"
    GT = "gt"
    GT_CORRECTED = "gt_corrected"
    VISITED = "visited"


class ColorMap(object):
    CM_VOLUME = 'viridis'

class BrainAge(object):
    ADULT = "Adult"
    P0 = "P0"
    E8p5 = "E8.5"
    E10p5 = "E10.5"
    E12p5 = "E12.5"
    E14p5 = "E14.5"
    E15p5 = "E15.5"
    E16p5 = "E16.5"
    E18p5 = "E18.5"


class ParamConst(object):
    NAME = 'name'

class AtlasConst(object):
    ADULT_MOUSE_ROOT_ID = 0
    DEV_MOUSE_ROOT_ID = 15564
    ALGN_INFO = 'align_info'
    ALGND_ON = "aligned_on"
    ALGND_JSON = "alignment_json"
    RS_ALGND_JSON = "rescaled_alignment_json"
    DS_TIF = "downsampled_tif"
    ANN_TIF = "ann_tif"
    FLIP_X = "flip_x"
    FLIP_Y = "flip_y"
    FLIP_Z = "flip_z"
    XIDX = "xidx"
    YIDX = "yidx"
    ZIDX = "zidx"
    REGION = "Region"
    ANNOTATION = "annotation"
    AUTOFLUORESCENCE = "autofluorescence"


class MetaData(object):
    CC_NPY = "cc_npy"
    CC_CSV = "cc_csv"
    CLIM = "clim"
    CCD_PHATHOM_PARAM_JSON = "ccd_phathom_param_json"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    STD = "std"
    TIF_REL_PATH = "tif_rel_path"
    ZARR_REL_PATH = "zarr_rel_path"
    AGE = "age"
