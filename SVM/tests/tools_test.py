from unittest import TestCase
import tools

class Test(TestCase):
    def test_orient(self):
        p, q, r = (0.11, 0.11), (0.11, 0.42), (0,1)
        self.assertEqual(tools.orient(p, q, r), tools.COUNTER_CW)
        self.assertEqual(tools.orient(q, p, r), tools.CLOCKWISE)

    def test_BinarySearcher(self):
        self.find_all_binary(list(range(11)), 0, 11, 0)
        self.find_all_binary(list(range(11)), 1, 10, 1)
        self.find_all_binary(list(range(2,9)), 2, 9, 2)
        self.find_all_binary(list(range(2, 9)), 3, 7, 3)
        self.find_all_binary(list(range(2, 9)), 3, 4, 4)
        self.find_all_binary(list(range(3, 3)), 3, 3, 5)

    def find_all_binary(self, l, mini, maxi, testid):
        for i in range(mini,maxi):
            testl = len(l)
            bo = tools.BinarySearcher(mini, maxi, discrete=True)
            n = -1
            while testl > 0:
                testl -= 1
                n = bo.next()
                if n == i:
                    break
                bo.feedback(higher=i>n)
            self.assertEqual(n, i, f"Failed on mini:{mini}, maxi:{maxi} on [{l[0]}, {l[-1]}), test-id {testid}")
