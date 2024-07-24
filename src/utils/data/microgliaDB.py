"""microgliaDB.py: Class to store microglia specific data information"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "01/31/2019"


class MDBLabel(object):
    """Microglia Dataset Label Definitions

    Ramified: Resting
    Amoeboid: Activated
    Garbage: to throw away
    """
    RAMIFIED = "Ramified"
    AMOEBOID = "Amoeboid"
    GARBAGE = "Garbage"
    UNCERTAIN = "Uncertain"
    lmap = {RAMIFIED:0, AMOEBOID: 1, GARBAGE: 2, UNCERTAIN: 3}

    def __init__(self):
        self.counter = {0: 0, 1: 0, 2: 0, 3: 0}

    def get_id(self, name):
        """with name of label, return id of label"""
        return self.lmap[name]

    def get_name(self, id_):
        """with id of label, return name of label"""
        return list(self.lmap.keys())[list(self.lmap.values()).index(id_)]


    def get_ordered_labels_by_id(self):
        return [y[0] for y in sorted(self.lmap.items(), key=lambda x: x[1])]

    def add(self, id_):
        """increase counter of id by 1"""
        self.counter[id_] += 1

    def subtract(self, id_):
        """decrease counter of id by one"""
        self.counter[id_] -= 1

    def reset_counter(self):
        """reset counter data"""
        self.counter = {0: 0, 1: 0, 2: 0, 3: 0}

    def get_total_count(self):
        """return total number of labels across the categories"""
        total = 0
        for key in self.counter.keys():
            total += self.counter[key]

        return total

    def get_count_msg(self):
        msg = ""
        total = self.get_total_count()
        for idx, key in enumerate(self.counter.keys()):
            number = self.counter[key]
            percentage = 0.0 if total == 0 else number*100./float(total)
            msg += "%s: %d (%.1f%%)"%(self.get_name(key)[0], number, percentage)
            if idx < len(self.counter) - 1:
                msg += ", "

        return msg
