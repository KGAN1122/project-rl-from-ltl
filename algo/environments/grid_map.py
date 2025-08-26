import numpy as np
# from IPython.display import display
from matplotlib import pyplot as plt


class GridMaps():
    """
    Stores the MDP environments and the corresponding tasks

    Attributes
    ----------
    shape : The shape of the MDP environment

    structure : The structure of the environment, including walls, traps, obstacles, etc.

    label : The label of each of the MDP states

    lcmap : The colour of different labels

    p : The probability that the agent moves in the intended direction, with probability (1-p)/2 of going sideways

    start : The start position in the MDP for the agent
    """

    def __init__(self):
        self.shape=None
        self.structure=None
        self.label=None
        self.lcmap=None
        self.p = None
        self.plot_start=None

    # def office_world(self):
    #     self.ltl='(G F t) & (G F l) & (G F a) & (G !o)'
    #     self.p=1

    #     # E: Empty, T: Trap, B: Obstacle
    #     self.structure=np.array([
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
    #     ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
    #     ])

    #     # Labels of the states
    #     self.label = np.array([
    #     [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
    #     [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),('t',),(),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),(),('l',),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),(),('a',),(),(),(),(),()],
    #     [(),('o',),(),(),(),(),(),(),(),('l',),(),(),(),('o',),()],
    #     [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
    #     [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
    #     [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
    #     ],dtype=object)
    #     # Colors of the labels
    #     self.lcmap={
    #         ('l',): 'orange',
    #         ('a',):'green',
    #         ('t',):'blue',
    #         ('o',):'red',
    #     }

    #     self.shape = self.structure.shape
    #     self.start = (9, 2)
    #     self.plot_start = (9, 2)
    #     self.name='office world'


    def office_world(self):
        self.ltl='(G F t) & (G F l) & (G F a) & (G !o)'
        # self.ltl='G((F t) & (F l) & (X F a)) & (G !o)'
        self.p=1

        # E: Empty, T: Trap, B: Obstacle
        self.structure=np.array([
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),('t',),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('o',),(),(),(),('a',),(),(),(),('l',),(),(),(),('o',),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
        ],dtype=object)
        # Colors of the labels
        self.lcmap={
            ('l',): 'orange',
            ('a',):'green',
            ('t',):'blue',
            ('o',):'red',
        }

        self.shape = self.structure.shape
        self.start = (9, 2)
        self.plot_start = (9, 2)
        self.name='office world'


    def office_world2(self):
        self.ltl='((F (l & (X ((G F t) & (G F a)))))|(F G b)) & (G !o)'
        # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
        self.p=0.5

        # E: Empty, T: Trap, B: Obstacle
        self.structure=np.array([
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'U', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['AU', 'U', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['B', 'E', 'B', 'B', 'B', 'E', 'B', 'B','B', 'E', 'B', 'B', 'B', 'E', 'B'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'R', 'E', 'E', 'E', 'E','E', 'E', 'E', 'E', 'E', 'E', 'E'],
        ['E', 'E', 'E', 'B', 'E', 'E', 'E', 'B','E', 'E', 'E', 'B', 'E', 'E', 'E']
        ])

        # Labels of the states
        self.label = np.array([
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),('t',),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [('b',),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),('o',),(),(),(),('a',),(),(),(),('l',),(),(),(),('o',),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),('t',),(),(),(),()],
        [(),(),(),(),(),('o',),(),(),(),('o',),(),(),(),(),()],
        [(),(),(),(),(),(),(),(),(),(),(),(),(),(),()]
        ],dtype=object)
        # Colors of the labels
        self.lcmap={
            ('l',): 'darkorange',
            ('a',):'green',
            ('t',):'blue',
            ('o',):'red',
            ('b',):'brown',
        }

        self.shape = self.structure.shape
        self.start = (9, 2)
        self.plot_start = (9, 2)
        self.name='office world 2'


    # def frozen_lake8x8(self):
    #     self.ltl = '((G F a)|(G F b)) & (G !h)'
    #     # self.ltl = '(!h) U (a | b)'
    #     # self.ltl = '((G F a) & (G F b)) & (G !h)'
    #     # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
    #     self.p = 0.34
    #     # self.p = 0.4

    #     # E: Empty, T: Trap, B: Obstacle P: slippery
    #     self.structure = np.array([
    #         ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'P', 'E', 'P', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
    #         ['P', 'E', 'E', 'P', 'P', 'P', 'E', 'P'],
    #         ['P', 'E', 'P', 'P', 'E', 'P', 'E', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'E'],
    #     ])

    #     # Labels of the states
    #     self.label = np.array([
    #         [(), (), (), (), (), (), (), ()],
    #         [(), (), (), (), (), (), (), ()],
    #         [(), (),    (),     ('h',), (), (), (), ()],
    #         [(), (),    (),     (),     (),('h',), (), ()],
    #         [(), (),    (),     ('h',), (), (), ('b',), ()],
    #         [(), ('h',), ('h',), (),     (), (), ('h',), ()],
    #         [(), ('h',), (),    (),  ('h',), (), ('h',), ()],
    #         [(), (), (), ('h',), (), (), (), ('a',)]
    #     ], dtype=object)
    #     # Colors of the labels
    #     self.lcmap = {
    #         ('h',): 'blue',
    #         ('a',): 'red',
    #         ('b',): 'darkorange',

    #     }

    #     self.shape = self.structure.shape
    #     self.start = (0, 0)
    #     self.plot_start = (0, 0)
    #     self.name = 'frozen lake'

    def frozen_lake8x8(self):
        self.ltl = '((G F b) & (G F a)) & (G !h)'
        # self.ltl = '(!h) U (a | b)'
        # self.ltl = '((G F a) & (G F b)) & (G !h)'
        # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
        self.p = 0.34
        # self.p = 0.4

        # E: Empty, T: Trap, B: Obstacle P: slippery
        self.structure = np.array([
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
            ['P', 'P', 'P', 'P', 'P', 'E', 'P', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
            ['P', 'E', 'E', 'P', 'P', 'P', 'E', 'P'],
            ['P', 'E', 'P', 'P', 'E', 'P', 'E', 'P'],
            ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'E'],
        ])

        # Labels of the states
        self.label = np.array([
            [(), (), (), (), (), (), (), ()],
            [(), (), (), (), (), (), (), ()],
            [(), (),    (),     ('h',), (), (), (), ()],
            [(), (),    (),     (),     (),('h',), (), ()],
            [(), (),    (),     ('h',), (), (), ('b',), ()],
            [(), ('h',), ('h',), (),     (), (), ('h',), ()],
            [(), ('h',), (),    (),  ('h',), (), ('h',), ()],
            [(), (), (), ('h',), (), (), (), ('a',)]
        ], dtype=object)
        # Colors of the labels
        self.lcmap = {
            ('h',): 'blue',
            ('a',): 'red',
            ('b',): 'darkorange',

        }

        self.shape = self.structure.shape
        self.start = (0, 0)
        self.plot_start = (0, 0)
        self.name = 'frozen lake'

    # def frozen_lake8x8(self):
    #     # self.ltl = '((G F a)|(G F b)) & (G !h)'
    #     self.ltl = '((G F b) & (G F a)) & (G !h)'
    #     # self.ltl = '(G F a) & (G !h)'
    #     # self.ltl='(F (l & (X ((G F t) & (G F a))))) & (G !o)'
    #     self.p = 0.34
    #     # self.p = 0.4

    #     # E: Empty, T: Trap, B: Obstacle P: slippery
    #     self.structure = np.array([
    #         ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
    #         ['P', 'P', 'P', 'P', 'E', 'E', 'P', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'P', 'P'],
    #         ['P', 'E', 'E', 'P', 'P', 'P', 'E', 'P'],
    #         ['P', 'E', 'P', 'P', 'E', 'E', 'E', 'P'],
    #         ['P', 'P', 'P', 'E', 'P', 'P', 'E', 'E'],
    #     ])

    #     # Labels of the states
    #     self.label = np.array([
    #         [(), (), (), (), (), (), (), ()],
    #         [(), (), (), (), (), (), (), ()],
    #         [(), (),    (),     ('h',), (), (), (), ()],
    #         [(), (),    (),     (),     ('b',),('h',), (), ()],
    #         [(), (),    (),     ('h',), (), (), (), ()],
    #         [(), ('h',), ('h',), (),     (), (), ('h',), ()],
    #         [(), ('h',), (),    (),  (), (), ('h',), ()],
    #         [(), (), (), ('h',), (), (), (), ('a',)]
    #     ], dtype=object)
    #     # Colors of the labels
    #     self.lcmap = {
    #         ('h',): 'blue',
    #         ('a',): 'red',
    #         ('b',): 'darkorange',

    #     }

    #     self.shape = self.structure.shape
    #     self.start = (0, 0)
    #     self.plot_start = (0, 0)
    #     self.name = 'frozen lake'
