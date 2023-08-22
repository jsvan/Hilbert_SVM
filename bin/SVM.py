import hilbert
import numpy

class SVM:



    def fit(self, X, Y):
        self.check_XY(X,Y)




    def check_XY(self, X, Y):
        assert len(X) == len(Y)
        # all X's have same dimension
        assert len(set([len(x) for x in X])) == 1
        assert numpy.all([len(y) == 1 for y in Y])








"""

TODO: given Linear seperable pointset
    Looking for a line l such that:
        maximize the minimum distance btwn point p to l
        
    claim by general optimization principals, in the euclidean space, think of this as the thickest road btwn points
    what conditions? Contact support vectors. If it only contacts two points, then you can rotate to make even wider
    
    color support vectors will be rotating
    DOes local maximality imply global maximality
    
    In hilbert distance isn't always perpenticular
    Build function to maximize ball around line
    1) Define hilbert ball about line
    2) where distance = 0.5*ln(cross ratio) cross ratio is (p,q; P, Q), where capitals are border
    3) 
    
    
    AS INPUT YOU WILL NEED VERTICES OF OMEGA SPACE 
    VERTICES MUST BE SORTED COUNTERCLOCKWISE ORDER
    
    Hilbert ball around point:
    This is equivalent to circle. Therefore given arbitrary point, you can say whether it's inside or outside. 
    
    1) Draw a line from EVERY VERTEX through p till it hits the boundary
        1a) You get [m (# sides)] 2 * m spokes
        1b) Every spoke defines a region
        1c) Pick a point q inside that region
        1d) The region's Omega boundaries (one on each side of omega), then you can extend those boundaries to the 
            vanishing point OUTSIDE of omega
        1e) THEN, the vanishing point vertex, if you extend a ray back into the regions, intersecting the point of 
            distance r. (If you don't choose point q, then you can perform a trig function to find the angle of line from
            vanishing point) 
        1f) Once you get the first line segment, you can continue it in other regions, with continuous but angular line 
            segments which intersect with the new vanishing points for every region
        
    
    Hilbert ball around line:
    
    2) the hilbert ball around q is tangent to the line l
        2a) p is the closest point to q on l iff:
        
            2aa) let P & Q be the points where pq hits the boundary of OMEGA (derivative symbol)
                and P is on l
            2ab) there exists supporting lines for OMEGA through P & Q that meet on l, ie, the vanishing point of P and Q 
                tangent lines is on the same point as l
        2b) p to q distance r is the radius of ball (metric ball)
        
        be ant walking along edge, connecting yourself to closest point p on l
        be ant on l (p) , where is your associated q line? 
        2c) expand 
    
    
    
    
    DISTANCE FROM POINT to LINE
    find closest place from l to q
    Vertice list is a circular list assume
    edges are Vi_Vi+1 (mod m)
    pick q, draw triangle in opposite verteces, expand tangent edges to meet at t
    one edge will do the job
    Let t be the point on l hit by extension of Vi_Vi+1
    once you know where t is
    
    find edge which intersects with l
    draw a line segment to intersect l (DONE by axiom)
    does it contain q? (NO, it's a line)
    continue lines from neighbors to intersect l, draw triangle OUTSIDE of OMEGA boundaries
    does it contain q? (NO!)
    step along verteces, draw triangle with those who intersect l
    until you find one that contains q
    draw a ray from point to vertex crossing l
    or vertex to p
    
    
    How do you know which lines intersect l? O(N?)
        can do in log^2 time, because can see what side of the point you are. 
            One part of biary search guesses edges, the other side of binary guesses tangency
    How do you know a point is in triangle?
    
    how do you represent a line?
    GRAPHICS Gems, Applications of linear algebra:
        Suppose l was given by some line equation, y=ax + b
        you have verteces vi - (xi, yi)
        vector u is vi+i - vi = (xi+1 - xi, yi+1 - yi)
        use stardard geometric functions
        if you have a starting point p, vector u, then the ray can describe any points:
            t is a real number here, u is a vector, p is starting point:
            w(t) = p + t * u
        interseect ray with line l:
            vi_vi+1
            set v (x values) to point on line's x
            set v (y values) to point on line's y
            
    Determine where support line hits from t to OMEGA space
    cute way to check. 3 points can be oriented in 3 ways, colinear, counter clockwise, clockwise.
     orientation test:
        orient(p, q, r) = {+1 if counterclockwise p-q-r, -1 if clockwise, 0 otherwise}
        Return the sign of the determinate of a the matrix
          
        sign(det(   
            {   1   px  py
                1   qx  qy
                1   rx  ry  }
                
        so if the line lands as tangent on top, test is clockwise for t, vi, vi-1 and t, vi, vi+1
        if the tests for both come out as same, then tangent, otherwise going through
        
    
    
    
    
    FOR 2 DIMENSIONS:
        HILBERT BALL around POINT has comp complexity of 
        
    
    
    Look at book by O'Rourke computational geometry in C
    
    CGAL coding library
    computational algorithmic library
        designed for accuracy adds precision with floating point filter
        slow for big applications, but should be good for hilbert geometry
        
    
    
    
    
    
    
    
    MEETING Friday 9:00am
    
        
    

"""

import numpy as np


class SVM:
    def __init__(self, C=1.0, tol=0.01, max_iter=100):
        self.C = C  # Regularization parameter
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Maximum number of iterations
        self.alphas = None  # Lagrange multipliers
        self.b = 0  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        self.b = 0

        iteration = 0
        while iteration < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):
                error_i = self._decision_function(X[i]) - y[i]
                if (y[i] * error_i < -self.tol and self.alphas[i] < self.C) or \
                        (y[i] * error_i > self.tol and self.alphas[i] > 0):
                    j = self._select_random_other_than(i, n_samples)
                    error_j = self._decision_function(X[j]) - y[j]

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue

                    alpha_j_new = alpha_j_old - (y[j] * (error_i - error_j)) / eta
                    if alpha_j_new > H:
                        alpha_j_new = H
                    elif alpha_j_new < L:
                        alpha_j_new = L

                    if abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)

                    b1 = self.b - error_i - y[i] * (alpha_i_new - alpha_i_old) * np.dot(X[i], X[i]) - \
                         y[j] * (alpha_j_new - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.b - error_j - y[i] * (alpha_i_new - alpha_i_old) * np.dot(X[i], X[j]) - \
                         y[j] * (alpha_j_new - alpha_j_old) * np.dot(X[j], X[j])

                    if 0 < alpha_i_new < self.C:
                        self.b = b1
                    elif 0 < alpha_j_new < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.alphas[i] = alpha_i_new
                    self.alphas[j] = alpha_j_new
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                iteration += 1
            else:
                iteration = 0

    def _decision_function(self, X):
        return np.dot(self.alphas * self.y, np.dot(self.X, X)) + self.b

    def _select_random_other_than(self, i, n):
        j = i
        while j == i:
            j = np.random.randint(0, n)
        return j


# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([-1, -1, 1, 1, 1])

svm = SVM(C=1.0, tol=0.01, max_iter=100)
svm.fit(X, y)
print("Alphas:", svm.alphas)
print("Bias:", svm.b)