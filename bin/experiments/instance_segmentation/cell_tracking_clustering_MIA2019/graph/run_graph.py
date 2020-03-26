
import queue
from graph.node import Node


def run_graph(fetches, feed_dict=None):
    """
    Function that runs a graph for given Nodes. For every Node, its parents are evaluated before
    such that every Node in the graph is calculated exactly once.
    Pre-calculated values for a Node in the graph may be given with feed_dict. In this case, neither the given Node,
    nor its parents are being calculated.
    TODO: currently all Node values are cached, which could increase memory consumption. Implement deletion of calculated objects when they are not needed anymore.
    :param fetches: List of Nodes.
    :param feed_dict: Dictionary of pre-calculated Nodes.
    :return: The calculated objects for all fetches (in the same order).
    """
    current_fetches = {}
    if feed_dict is not None:
        current_fetches.update(feed_dict)

    # create Lifo queue and add current fetches list.
    node_queue = queue.LifoQueue()
    for fetch in fetches:
        node_queue.put(fetch)

    while not node_queue.empty():
        current_node = node_queue.get()
        assert isinstance(current_node, Node), 'The current node is not a Node object. Either set its value via feed_dict or fix the graph. current_node = ' + str(current_node)

        # check if parents are already calculated
        parents = list(current_node.parents) + list(current_node.kwparents.values())
        all_parents_calculated = True
        for parent in parents:
            if parent not in current_fetches:
                # if current parent is not calculated,
                # add the current node again (only once) to the queue
                if all_parents_calculated is True:
                    node_queue.put(current_node)
                    all_parents_calculated = False
                # add parent as next node to the queue
                node_queue.put(parent)

        if all_parents_calculated is False:
            continue

        parents_values = [current_fetches[parent] for parent in current_node.parents]
        kwparents_values = dict([(parent_key, current_fetches[parent]) for parent_key, parent in current_node.kwparents.items()])
        current_output = current_node.get(*parents_values, **kwparents_values)
        current_fetches[current_node] = current_output

    fetches_outputs = [current_fetches[fetch] for fetch in fetches]

    return fetches_outputs
