
from __future__ import print_function

import unittest

import numpy

import logging
from aotools.functions import pupil

# Get the logger:
logging.basicConfig()
# Create a LOGGER:
log = logging.getLogger(__name__)
# Set the logging LEVEL: (filter??)
log.setLevel(2)

class TestCircle(unittest.TestCase):
    def test_circle( self ):

        log.info("Create circles with (0,5), (1,5), (2,5), (0,6),...,(3,6)")
        # Define the expected outputs:
        # >>> circle.circle(0, 5)
        ce = numpy.array([[ 0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.]])
        c = pupil.circle(0, 5)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(1, 5)
        ce = numpy.array([[ 0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  1.,  1.,  1.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.]])
        c = pupil.circle(1, 5)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(2, 5)
        ce = numpy.array([[ 0.,  0.,  1.,  0.,  0.],
                          [ 0.,  1.,  1.,  1.,  0.],
                          [ 1.,  1.,  1.,  1.,  1.],
                          [ 0.,  1.,  1.,  1.,  0.],
                          [ 0.,  0.,  1.,  0.,  0.]])
        c = pupil.circle(2, 5)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(0, 6)
        ce = numpy.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.]])
        c = pupil.circle(0, 6)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(1, 6)
        ce = numpy.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  1.,  1.,  0.,  0.],
                          [ 0.,  0.,  1.,  1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.]])
        c = pupil.circle(1, 6)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(2, 6)
        ce = numpy.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  0.,  1.,  1.,  0.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  1.,  1.,  1.,  1.,  0.],
                          [ 0.,  0.,  1.,  1.,  0.,  0.],
                          [ 0.,  0.,  0.,  0.,  0.,  0.]])
        c = pupil.circle(2, 6)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        #  circle.circle(3, 6)
        ce = numpy.array([[ 0.,  1.,  1.,  1.,  1.,  0.],
                          [ 1.,  1.,  1.,  1.,  1.,  1.],
                          [ 1.,  1.,  1.,  1.,  1.,  1.],
                          [ 1.,  1.,  1.,  1.,  1.,  1.],
                          [ 1.,  1.,  1.,  1.,  1.,  1.],
                          [ 0.,  1.,  1.,  1.,  1.,  0.]])
        c = pupil.circle(3, 6)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )


        # This is a crucial test since this is used in the dragon config file:
        # circle(15.5, 31) - circle(3.7, 31)
        ce = numpy.array(
            [[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
             [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]])
        c = pupil.circle(15.5, 31) - pupil.circle(3.7, 31)
        self.assertEqual( c.tolist(), ce.tolist() )
        self.assertTrue( (c == ce).all() )

        log.info("Create circles with (2,6,(0.5,0.5)), (1,5,(0.5,0.5)), etc.")
        c1 = pupil.circle(2, 6, (0.5,0.5))[1:,1:]
        c2 = pupil.circle(2, 5)
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )

        c1 = pupil.circle(1, 6, (0.5,0.5))[1:,1:]
        c2 = pupil.circle(1, 5)
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )

        c1 = pupil.circle(0, 6, (0.5,0.5))[1:,1:]
        c2 = pupil.circle(0, 5)
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )

        c1 = pupil.circle(0, 5, (0.5, 0.5))[1:, 1:]
        c2 = pupil.circle(0, 6)[1:-1, 1:-1]
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )

        c1 = pupil.circle(1, 5, (0.5, 0.5))[1:, 1:]
        c2 = pupil.circle(1, 6)[1:-1, 1:-1]
        #c2[3,2] = 3.0 # to test the testing
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )

        c1 = pupil.circle(2, 5, (0.5, 0.5))[1:, 1:]
        c2 = pupil.circle(2, 6)[1:-1, 1:-1]
        self.assertEqual( c1.tolist(), c2.tolist() )
        self.assertTrue( (c1 == c2).all() )


        log.info("Raise TypeError if inputs not of correct type")
            #pylab.imshow(c-c1, interpolation="nearest")
