
X, Y = 0, 1


def det(a, b, c, d):
    return (a*d) - (b*c)

"""
TODO? Maybe not needed
def line_from_two_points(a, b):
    [(v, lambda t: p.coords + t * (p.coords - np.array(v))) for v in self.vertices]
    rise = b[Y] - a[Y]
    run = b[X] - a[X]
    if run == 0:
        
    slope = rise/run
"""



def intersect(line_a, line_b):
    """
    :line_a: iterable of two points defining a line
    :line_b: iterable of two points defining a line
    using determinants (following code). Better way?
    https://mathworld.wolfram.com/Line-LineIntersection.html
    """

    (x1, y1), (x2, y2) = line_a
    (x3, y3), (x4, y4) = line_b

    xy12 = det(x1, y1, x2, y2)
    xy34 = det(x3, y3, x4, y4)
    x1m2 = x1 - x2
    x3m4 = x3 - x4
    y1m2 = y1 - y2
    y3m4 = y3 - y4
    denominator = det(x1m2, y1m2, x3m4, y3m4)

    x = det(xy12, x1m2, xy34, x3m4) / denominator
    y = det(xy12, y1m2, xy34, y3m4) / denominator
    return x, y








    # start up
    # pipelines, models
    # engineering
    # base salary
    # remote
    # focussed computer science phd
    # nlp research
    # foundation model for business data manufacturing supply data
    # meta nlp specialist
    # llm research ml side
    # stable manufacturing 90% business is email, spreadsheet is no code, allows supply chain to build own products,
    # cant take on more customers, four year sas model, good guys, 2nd data scientist, meeting all data scientists next steps
    # amirsina torfi
    # 6 seattle 6 nyc 6 in san fran
    # llm to read emails, what managers would do in specific situations