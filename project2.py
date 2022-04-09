#############################################################################################################
# COM S 476/576 Project 2 Solution
# Donald Calhoun
# Built off of Solutions from Project 1 and Homework 1 with code written by Tichakorn Wongpiromsarn
#############################################################################################################

import json, sys, argparse, math
import numpy as np
from queue import QueueBFS
from shapely.geometry import Polygon


def get_link_positions(config, W, L, D):
    """Compute the positions of the links and the joints of a 2D kinematic chain A_1, ..., A_m

    @type config: a list [theta_1, ..., theta_m] where theta_1 represents the angle between A_1 and the x-axis,
        and for each i such that 1 < i <= m, \theta_i represents the angle between A_i and A_{i-1}.
    @type W: float, representing the width of each link
    @type L: float, representing the length of each link
    @type D: float, the distance between the two points of attachment on each link

    @return: a tuple (joint_positions, link_vertices) where
        * joint_positions is a list [p_1, ..., p_{m+1}] where p_i is the position [x,y] of the joint between A_i and A_{i-1}
        * link_vertices is a list [V_1, ..., V_m] where V_i is the list of [x,y] positions of vertices of A_i
    """

    if len(config) == 0:
        return ([], [])

    joint_positions = [np.array([0, 0, 1])]
    link_vertices = []

    link_vertices_body = [
        np.array([-(L - D) / 2, -W / 2, 1]),
        np.array([D + (L - D) / 2, -W / 2, 1]),
        np.array([D + (L - D) / 2, W / 2, 1]),
        np.array([-(L - D) / 2, W / 2, 1]),
    ]
    joint_body = np.array([D, 0, 1])
    trans_mat = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    for i in range(len(config)):
        a = D if i > 0 else 0
        trans_mat = np.matmul(trans_mat, get_trans_mat(config[i], a))
        joint = np.matmul(trans_mat, joint_body)
        vertices = [
            np.matmul(trans_mat, link_vertex) for link_vertex in link_vertices_body
        ]
        joint_positions.append(joint)
        link_vertices.append(vertices)

    return (joint_positions, link_vertices)


def get_trans_mat(theta, a):
    """Return the homogeneous transformation matrix"""
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), a],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="C-Space and C-Space Obstacles:")
    parser.add_argument(
        "desc",
        metavar="problem_description_path",
        type=str,
        help="path to the problem description file containing the obstacles, width of link, length of link, distance between, the initial cell, and the goal region",
    )
    parser.add_argument(
        "--out",
        metavar="output_path",
        type=str,
        required=False,
        default="",
        dest="out",
        help="path to the output file",
    )

    args = parser.parse_args(sys.argv[1:])

    return args

def parse_desc(desc):
    """Parse problem description json file to get the problem description"""
    with open(desc) as desc:
        data = json.load(desc)

    O = data["O"]
    W = data["W"]
    L = data["L"]
    D = data["D"]
    xI = tuple(data["xI"])
    XG = [tuple(x) for x in data["XG"]]
    U = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    return (O, U, W, L, D, xI, XG)

"""
Function that takes the obstacles as a list, the current Graph, width x length x Dist of each polygon
then checks if there is a collision and changes that part of graph to 1
"""
def check_collision(O, G, W, L, D):
    # Iterates thru every position on G
    for r in range(len(G)):
        for c in range(len(G[r])):
            #Gets the robots configuration
            initRobotPos = list()
            initRobotPos.append((r * math.pi) / 180)
            initRobotPos.append((c * math.pi) / 180)
            (joint_positions, link_vertices) = get_link_positions(initRobotPos, W, L, D)

            # Makes polygons for the robots 2 links
            p1Vert = list()
            for i in link_vertices[0]:
                p1coords = (i[0], i[1])
                p1Vert.append(p1coords)
            p1 = Polygon([p1Vert[0], p1Vert[1], p1Vert[2], p1Vert[3]])
            p2Vert = list()
            for i in link_vertices[1]:
                p2coords = (i[0], i[1])
                p2Vert.append(p2coords)
            p2 = Polygon([p2Vert[0], p2Vert[1], p2Vert[2], p2Vert[3]])

            # Creates polygons for each obstacle and checks for collision with robot
            for i in O:
                vertices = list()
                for j in i:
                    coords = (j[0], j[1])
                    vertices.append(coords)
                p3 = Polygon([vertices[0], vertices[1], vertices[2], vertices[3]])
                if p1.intersects(p3) or p2.intersects(p3):
                    G[c][r] = 1
    return G

def fsearch(G, U, xI, XG, Q):
    """Return the list of visited nodes and a path from xI to XG based on the given algorithm

    This is the general template for forward search describe in Figure 2.4 in the textbook.

    @type G: 2-dimensional list such that G[j][i] indicates whether cell (i,j) is occupied
    @type U: a list of tuples specifying all the possible actions
    @type xI: a tuple specifying the initial cell
    @type XG: a list of tuples specifying the goal region
    @type Q: a Queue object with insert, pop, get_visited, and get_path functions

    @return: a tuple (visited_nodes, path) where visited_nodes is the list of
    nodes visited by bfs and path is a path from cell xI to a cell in XG
    """

    def is_free(x):
        """Determine whether cell x is free"""
        return (
            x[0] >= 0
            and x[0] < len(G[0])
            and x[1] >= 0
            and x[1] < len(G)
            and G[x[1]][x[0]] < 0.5
        )

    def Ux(x):
        """Return the action space for cell x"""
        return [u for u in U if is_free(f(x, u))]

    def f(x, u):
        """Return the new cell obtained by applying action u at cell x"""
        return (x[0] + u[0], x[1] + u[1])

    if not is_free(xI):
        return []

    Q.insert(xI, None)

    while len(Q) > 0:
        x = Q.pop()
        if x in XG:
            return Q.get_path(xI, x)
        for u in Ux(x):
            xp = f(x, u)
            Q.insert(xp, x)
    return []

if __name__ == "__main__":
    args = parse_args()
    (O, U, W, L, D, xI, XG) = parse_desc(args.desc)
    G = [[0] * 360 for _ in range(360)]
    G = check_collision(O, G, W, L, D)
    path = fsearch(G, U, xI, XG, QueueBFS())
    result = {"G": G, "path": path}
    with open(args.out, "w") as outfile:
        json.dump(result, outfile)