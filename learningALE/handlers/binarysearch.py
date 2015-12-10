import numpy as np


class Node:
    def __init__(self, value, extra_vals=None):
        self.left = None
        self.right = None
        self.value = value
        self.extra_vals = extra_vals

    def insert(self, new_val, extra_vals=None):
        if new_val >= self.value:
            if self.right:
                self.right.insert(new_val, extra_vals)
            else:
                self.right = Node(new_val, extra_vals)
        else:
            if self.left:
                self.left.insert(new_val, extra_vals)
            else:
                self.left = Node(new_val, extra_vals)

    def pop_max(self):
        if self.right:
            ret_val, extra_vals, hanging_left, term = self.right.pop_max()

            if term:
                del self.right
                self.right = hanging_left
            term = 0
        else:
            ret_val = self.value
            extra_vals = self.extra_vals
            hanging_left = self.left
            term = 1
        return ret_val, extra_vals, hanging_left, term

    def get_size(self):
        raise NotImplementedError

    def get_xy_vals(self, node_yx_vals, loc, depth, curr_depth=0):
        node_yx_vals.append([curr_depth, loc - 1, self.value])
        horizontal_move = 2**(depth-(curr_depth+1)-1)
        if self.right:
            node_yx_vals = self.right.get_xy_vals(node_yx_vals, loc + horizontal_move, depth, curr_depth + 1)
        if self.left:
            node_yx_vals = self.left.get_xy_vals(node_yx_vals, loc - horizontal_move, depth, curr_depth + 1)
        return node_yx_vals

    def depth(self):
        if self.left:
            left_depth = self.left.depth()
        else:
            left_depth = 0
        if self.right:
            right_depth = self.right.depth()
        else:
            right_depth = 0
        return max(left_depth, right_depth) + 1


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, val, extra_vals=None):
        if self.root:
            self.root.insert(val, extra_vals)

        else:
            self.root = Node(val, extra_vals)

    def pop_max(self):
        val, extra_vals, hanging_left, term = self.root.pop_max()
        # if terminal node, we are popping the root, set new root to hanging_left
        if term:
            self.root = hanging_left
        return val, extra_vals

    def plot(self):
        depth = self.root.depth()
        node_yx_vals = list()
        yx_list = self.root.get_xy_vals(node_yx_vals, loc=2**(depth-1), depth=depth)
        yx_list = np.asarray(yx_list)
        import matplotlib.pyplot as plt
        # annotate text values http://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
        fig, ax = plt.subplots()
        ax.scatter(yx_list[:, 1], yx_list[:, 0]*-1, c=yx_list[:, 2], linewidths=0)
        for ind, val in enumerate(yx_list[:, 2]):
            ax.annotate('{:03.3f}'.format(val), (yx_list[ind, 1], yx_list[ind, 0]*-1))
        plt.show()


if __name__ == '__main__':
    root = Node(np.random.random())
    import time

    st = time.time()
    for add in range(500):
        b = np.random.random()
        root.insert(b)

    for add in range(1000):
        b = np.random.random()
        root.insert(b)
        root.pop_max()

    et = time.time()

    # test popping the top
    # root = Node(np.inf)
    # root.insert(0)
    # ret_vals, extra_vals, hanging_left, term = root.pop_max()
    # print(ret_vals, hanging_left, term)
    print(et-st)
    root.plot()

