"""
Based on lesson 35
"""
import numpy as np


class ProcessControl:
    """
    Control the process of the lane detection of a video
    """
    def __init__(self):
        # max errors
        self.maxErrors = 5
        # current errors. Starts in the maximun to force the full search for the first image
        self.nErrors = self.maxErrors

        # vars to save the last validated image params
        self.left_fit = np.zeros(3)
        self.right_fit = np.zeros(3)


    def validateImageResult(self, left_curvature, right_curvature):
        """
        Validate the result of one image
        The validation is simple:
        - if the one of the curvature angles are X times smaller than the other,
            the validation is correct, set the current errors to 0 and returns True
        - If not, add 1 to current errors and return False

        Returns boolean
        """
        curvatureDifferential = 5
        if left_curvature * curvatureDifferential < right_curvature or left_curvature > right_curvature * curvatureDifferential:
            self.nErrors = self.nErrors + 1
            return False
        else:
            self.nErrors = 0
            return True


    def processFromScratch(self):
        """
        Check the current errors number. 
        If the current errors are equeal or bigger than maximum errors, the process must start from scratch

        Returns boolean
        """
        if self.nErrors >= self.maxErrors:
            return True
        else:
            return False


    def setnErrors(self, nErrors):
        self.nErrors = nErrors

    def getLeftFit(self):
        return self.left_fit

    def setLeftFit(self, left_fit):
        self.left_fit = left_fit

    def getRightFit(self):
        return self.right_fit

    def setRightFit(self, right_fit):
        self.right_fit = right_fit
