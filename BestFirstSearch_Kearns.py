#Contributions
#Together: We both collaborated in class to finish the code.
#Ronan: Found the Task 1 and Task 2 answers.
#Samith: Wrote the comments.
import sys
from collections import deque

from utils import *


class Problem:
    
    #The constructor sets the starting state and sometimes a goal state, if there is only one. Additional arguments can be added by your subclass's constructor.
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal
        
    #This function returns a list of possible actions that can be taken in a given state. If there are many actions, it is recommended to use an iterator instead of building the entire list at once.
    def actions(self, state):
        raise NotImplementedError
        
    #This function takes in a state and an action and returns the resulting state after executing the given action in the given state. The action must be one of the possible actions that can be taken in the given state, which are obtained by calling the actions(state) function.
    def result(self, state, action):
        raise NotImplementedError

    #This function returns a boolean value indicating whether the given state is a goal state or not. By default, it checks if the state is equal to the self.goal attribute or if the state is in the self.goal list. If checking against a single goal is not sufficient, this function can be overridden to define a more customized goal-checking method.
    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    #This function calculates the cost of a solution path that starts at state1, applies an action, and arrives at state2, given the cost c to reach state1. If the problem does not require considering the path taken, the function only considers state2. However, if the path taken is relevant, the function may also consider state1 and action. By default, this function assumes that each step in the path has a cost of 1
    def path_cost(self, c, state1, action, state2):
        return c + 1

    #In optimization problems, every state is assigned a value, and algorithms such as Hill Climbing attempt to maximize this value.
    def value(self, state):
        raise NotImplementedError


class Node:

    #This code creates a new node in a search tree. The node is derived from a parent node by applying an action to it.
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
    
    #This functions returns a string representation of the object. Specifically, when called on an instance of the Node class, it returns a string in the format of "<Node state>", where state is the state attribute of the Node object. This function is typically used for debugging and printing information about the object.
    def __repr__(self):
        return "<Node {}>".format(self.state)

    #Defines the comparison of two Node objects by their state values.
    def __lt__(self, node):
        return self.state < node.state

    #Given a problem, returns a list of child nodes for the current node.
    def expand(self, problem):
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    #Creates a new Node with the state resulting from applying the given action to the current node's state. The new node is connected to the current node.
    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(
            self.path_cost, self.state, action, next_state))
        return next_node

    #Returns the list of actions taken to reach the current node's state from the root node's state.
    def solution(self):
        return [node.action for node in self.path()[1:]]

    #Returns a list of nodes representing the path from the root node to the current node.
    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))
     
    #Compares two Node objects for equality based on their state values.
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
        
    #Returns the hash value of the Node object based on its state value.
    def __hash__(self):
        return hash(self.state)

#A class that represents a graph, where nodes are connected by edges (links). Supports adding links between nodes with associated distances, as well as getting a list of nodes and their links.
class Graph:

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    #If the graph is directed, makes it undirected by adding edges in the opposite direction for each existing edge.
    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    #Adds a link from node A to node B with the given distance. If the graph is undirected, also adds a link from node B to node A.
    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    #Returns the distance between nodes a and b if b is specified, otherwise returns a dictionary of links from node a to its neighbors.
    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    #Returns a list of nodes in the graph.
    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values()
                 for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

#A function that returns an undirected Graph object, which is a Graph object with the directed flag set to False.
def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

#A class that represents a priority queue, where items are popped from the queue in order of priority (determined by a function f). Supports appending items to the queue and getting the size of the queue.
class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    #Insert item at its correct position.
    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    #Insert each item in items at its correct position.
    def extend(self, items):
        for item in items:
            self.append(item)
    #Pop and return the item (with min or max f(x) value) depending on the order.
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception
    #Return current capacity of PriorityQueue.
    def __len__(self):
        return len(self.heap)

    #Return True if the key is in PriorityQueue.
    def __contains__(self, key):
        return any([item == key for _, item in self.heap])

    #Returns the first value associated with key in PriorityQueue. Raises KeyError if key is not present.
    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    #Delete the first occurrence of key.
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

#The problem of searching a graph from one node to another.
class GraphProblem(Problem):

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    #The actions at a graph node are just its neighbors.
    def actions(self, A):
        return list(self.graph.get(A).keys())

    #The result of going to a neighbor is just that neighbor.
    def result(self, state, action):
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    #Find minimum value of edges.
    def find_min_edge(self):
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m
    #h function is straight-line distance from a node's state to goal.
    def h(self, node):
        
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf

#Search the nodes with the lowest f scores first. You specify the function f(node) that you want to minimize; for example,if f is a heuristic estimate to the goal, then we have greedy best first search; if f is node.depth then we have breadth-first search.There is a subtlety: the line "f = memoize(f, 'f')" means that the f values will be cached on the nodes as they are computed. So after doing a best first search you can examine the f values of the path returned.
def best_first_graph_search(problem, f, display=False):
 
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and",
                      len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return None


def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)


basic_map = UndirectedGraph(dict(S=dict(A=3, B=1, C=8), D=dict(A=3),
                                 E=dict(A=7),

                                 G=dict(A=15, B=20, C=5)))  # start state: S, goal state: G

basic_problem = GraphProblem('S', 'G', basic_map)  # print the solution path

print("Task 1: Basic Problem Solution Path")
print("Solution path:", uniform_cost_search(basic_problem).path())
#print("Solution path:", best_first_graph_search(basic_problem, lambda node: node.path_cost).path())


romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))

romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)

romania_locations = romania_map.locations
# print(romania_locations) with new line
print("Task 2: Romania Solution Path")
# do best_first_graph_search on the romania_problem
print("Solution path:", best_first_graph_search(romania_problem, lambda node: node.path_cost).path())
