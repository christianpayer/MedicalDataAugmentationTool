
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
    node_queue = []
    for fetch in fetches:
        if (feed_dict is None or fetch not in feed_dict) and (fetch not in node_queue):
            node_queue.append(fetch)

    while len(node_queue) > 0:
        current_node = node_queue.pop()
        assert isinstance(current_node, Node), 'The current node is not a Node object. Either set its value via feed_dict or fix the graph. current_node = ' + str(current_node)

        # check if parents are already calculated
        all_parent_nodes = list(current_node.get_parents()) + list(current_node.get_kwparents().values())
        all_parents_calculated = True
        for parent in all_parent_nodes:
            if parent not in current_fetches:
                # if current parent is not calculated,
                # add the current node again (only once) to the queue
                if all_parents_calculated is True:
                    node_queue.append(current_node)
                    all_parents_calculated = False
                # if parent is somewhere in node_queue, delete it such that it will not be calculated twice
                if parent in node_queue:
                    node_queue.remove(parent)
                # add parent as next node to the queue
                node_queue.append(parent)

        if all_parents_calculated is False:
            continue

        # check if the only parent node is a Node
        if len(all_parent_nodes) == 1 and isinstance(current_fetches[all_parent_nodes[0]], Node):
            # replace current parent node with parents of all_parent_node
            # this is very hacky, but allows to incorporate conditions into the graph
            only_parent_node = current_fetches[all_parent_nodes[0]]
            parents_values = [current_fetches[parent] for parent in only_parent_node.get_parents()]
            kwparents_values = dict([(parent_key, current_fetches[parent]) for parent_key, parent in only_parent_node.get_kwparents().items()])
        else:
            # set parents to fetched objects
            parents_values = [current_fetches[parent] for parent in current_node.get_parents()]
            kwparents_values = dict([(parent_key, current_fetches[parent]) for parent_key, parent in current_node.get_kwparents().items()])
        current_output = current_node.get(*parents_values, **kwparents_values)
        current_fetches[current_node] = current_output

        # if current_output is a Node object, put it into the node_queue as it probably needs to be processed
        if isinstance(current_output, Node):
            node_queue.append(current_output)

    fetches_outputs = [current_fetches[fetch] for fetch in fetches]

    return fetches_outputs
