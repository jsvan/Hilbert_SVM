import numpy as np
from math import log
from numpy.linalg import norm
import euclidean
import tools


def nielson_dist(p, q):
   if np.allclose(p, q): return 0  # np.allclose returns true if all dim are within EPSILON
   idx = np.logical_not(np.isclose(p, q))  # np.isclose returns an array of bool, for each dimension if within EPSILON
   if (idx.sum() == 1): return 0  # returns 0 if only ONE dimension is not close (??)
   lamb = p[idx] / (p[idx] - q[idx])  # does dimension-wise arithmetic on those dimensions which are not within epsilon.
   t0 = lamb[lamb <= 0].max()  # t0 = resulting max element less than 0
   t1 = lamb[lamb >= 1].min()  # t1 = resulting min element bigger than 1
   if np.isclose(t0, 0) or np.isclose(t1, 1): return np.inf  # From here on idk
   return np.abs(np.log(1 - 1 / t0) - np.log(1 - 1 / t1))

def dist_with_boundary_intersections(p, q, A, B):
   """
   A--p--q--B
      |-pB--|
         |qB|
   |-qA--|
   |pA|
   :param p:
   :param q:
   :param b1:
   :param b2:
   :return:
   """
   pB = norm(p - B)
   qB = norm(q - B)
   qA = norm(q - A)
   pA = norm(p - A)
   crossratio = (pB / qB) * (qA / pA)
   return abs(log(crossratio))


def get_euclidean_midway_point_from_hdist(p, q, hdist, A, B):
   """
   A--p--q--B
   Will find point on line between p and q, which is hdist from both
   :param p:
   :param q:
   :param hdist:
   :param A:
   :param B:
   :return:
   """
   def newpoint(t): return p + (q-p) * t
   # bsearch will be granting us t values
   bsearch = tools.BinarySearcher(0, 1, discrete=False)
   t = bsearch.next()
   point = newpoint(t)
   dist = dist_with_boundary_intersections(point, q, A, B)
   while not np.isclose(dist, hdist):
      if dist > hdist:  # point is closer to p than to q
         bsearch.feedback(higher=True)
      else:
         bsearch.feedback(higher=False)
      t = bsearch.next()
      point = newpoint(t)
      dist = dist_with_boundary_intersections(point, q, A, B)
   return point



def get_boundary_intersections(p, q, boundaries):
   """
   :param p:
   :param q:
   :param boundaries: list of two boundaries. Each boundary is a list of two points.
   :return: points A and B, of order A--p--q--B
   """
   A, B = euclidean.intersect([p, q], boundaries[0]), \
      euclidean.intersect([p, q], boundaries[1])
   # This is if A and B are swapped, which is handy because my discovery algorithm doesn't handle orientation for me
   # even though ironically it uses something called the orient() algorithm a dozen times ...
   pB = norm(p - B)
   qB = norm(q - B)
   if qB > pB:
      A, B = B, A
   return A, B



class HilbertianHodgePodge:
   """
   This exists because I dont know how to organize code. Send help.
   """

   def __init__(self, p, q, omega):
      self.p, self.q = p, q
      self.omega = omega
      self.midpoint = None
      self.boundaries = None
      # between p and q
      self.hdist = None
      # boundary intersections
      self.A, self.B = None, None
      self.spokes = None


   def get_boundaries(self):
      if self.boundaries is None:
         self.boundaries = self.omega.find_boundaries_of_line(self.p, self.q)
      return self.boundaries


   def get_hdist(self):
      """
      hdist between two points p and q
      :return: float
      """
      if self.hdist is None:
         self.hdist = dist_with_boundary_intersections(self.p, self.q, *self.get_boundary_intersections())
      return self.hdist
   
   
   # gets hilbertian midpoint between p and q
   def get_midpoint(self):
      if self.midpoint is None:
         self.midpoint = get_euclidean_midway_point_from_hdist(self.p, self.q, self.get_hdist() / 2, *self.get_boundary_intersections())
      return self.midpoint
   
   
   def get_boundary_intersections(self):
      if self.A is None or self.B is None:
         self.A, self.B = get_boundary_intersections(self.p, self.q, self.get_boundaries())
      return self.A, self.B
   
   
   def get_best_dividing_line(self):
      vanishing_point = euclidean.intersect(*self.get_boundaries())
      return [self.get_midpoint(), vanishing_point]
   
   
   
class Omega:

   def __init__(self, vertices=[[0, 0], [0, 1], [1, 0]]):
      self.vertices = vertices
      # This is a hack for binary search, wraps around the final points. Whateverr
      self.vertices_expanded = [vertices[-1]] + vertices + [vertices[0]]


   def spokes(self, p):
      """
      :param p: np array point in omega
      :return: list of tuples: (coords of the omega vertex, lambda equation for related spoke)
      """
      return [(v, lambda t: p.coords + t * (p.coords - np.array(v))) for v in self.vertices]


   def find_boundaries_of_line(self, p, q):
      """
      Run orientation tests to find intersections with boundaries. Log time search.
      :param p:
      :param q:
      :return: two lines, each of two points.
      """
      def find_boundary(p,q, v):
         # :param top: There are two boundary lines to find. True/False for which of the two to look for.
         numvert = len(v)
         binarysearch = tools.BinarySearcher(0, numvert, discrete=True)
         attempts = numvert + 1

         while attempts > 0:
            attempts -= 1
            leftidx = binarysearch.next()
            leftv, rightv = v[leftidx], v[(leftidx + 1) % numvert]

            if tools.orient(p, q, leftv) == tools.COUNTER_CW:  # BAD ORIENTATION
               binarysearch.feedback(higher=False)
               continue
            if tools.orient(p, q, rightv) == tools.CLOCKWISE:
               binarysearch.feedback(higher=True)
               continue
            return (leftv, rightv)
         raise Exception("Finding boundary with binary search failed")

      return find_boundary(p, q, self.vertices_expanded), find_boundary(q, p, self.vertices_expanded)













"""
* the line is just different from the point with maximum clearence
* Look for just a line with maximum hilbert 
* Grow a ball at p and q
* at some point they will
* so grow hilbert ball around spokes
* essentially taking minimum(radius(of bisecting line))
* will probably be point to edge

* ordering spokes:
   ** pick reference direction (x axis) then compute a collection of angles with complements + pi
   ** orientation test _v7_ falls between v2, v3 iff           # avoids floating point rounding
         orientation(v7, p, v2) != orientation(v7, p, v3)
         




Once you find the midpoint, take all vertices + vanishing points, draw spokes through intersection point. The middle sectors will be the possibility for the euc line

by triangle inequality 

need to do binary search around edges to find where AB intersects boundaries, find the boundaries
Then you can calculate the Hdist and solve for median point between given normal hdist = ln (cross entropy)
Once you find the right edge, you know the correct spokes 
   take vanishing point of two boundary edges, then connect midpoint wiht vanishing point, the intersection points are of that segment.
   Because the ball is convex, can simply connect that vanishing point with that midpoint.




Geodesics are straight lines: So euclidean line pq, with equidistance in hilbert sense point is midpoint.
There is a sector which contains minimum point
It is always a straight line when only 2 edges involved
If you extend the line outside of the sector, it is a candidate for the best line
   We know its a bisector, so we know both hilbert balls around p, q touch this line
   if we extend this line, neither ball will intersect it again, unless its on an edge
   by convexity, it does not go inside the ball closer to p or q, therefore is a correct dividing line
   
Next exercise:
   3 points
   Solve in poly time, with right answer (prove?)
   Solve in poly log time
   Solve in log^m(n)
   meeting 9am friday 25th or so
   
   
entering points myself for convex:
   

"""







