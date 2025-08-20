from dataclasses import dataclass

@dataclass
class Node:
    def __init__(self, value, prev, next):
        self.value = value
        self.prev = prev
        self.next = next

@dataclass
class DoubleLinkedListWord:
    def __init__(self, word):
        self.word = word
        self.head = Node(None, None, None)
        self.tail = Node(None, self.head, None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.position_map = {}
        for character in word:
            self.append(character)


    # Return appended node
    def append(self, value):
        last_node = self.tail.prev
        new_node = Node(value, prev=last_node.prev, next=self.tail)
        last_node.next = new_node
        self.tail.prev = new_node
        new_node.prev = last_node
        new_node.next = self.tail
        if value not in self.position_map:
            self.position_map[value] = []
        self.position_map[value].append(new_node)
        return new_node

    # Delete two merged node, return the node before the first node. 
    # So later we can call insert with this position. 
    # If resource we want to delete doesn't exist, return null
    def delete(self, value_first, value_second):
        # Make sure delete entities exist
        if value_first not in self.position_map or value_second in self.position_map:
            if value_first not in self.position_map:
                print("Word: " + self.word + " ,delete node failed, first node doesn't exist, first node value: " + value_first)
            else:
                print("Word: " + self.word + "delete node failed, second node doesn't exist, second node value: " + value_second)
            return None
        node_first = self.position_map[value_first]
        node_second = self.position_map[value_second]
        if node_first.next != node_second or node_second.prev != node_first:
            print("Word: " + self.word + "delete node failed, nodes are not adjacent, first node value: " + value_first + ", second node value: " + value_second)
            return None
        # Deletion
        node_before_first = node_first.prev
        node_after_second = node_second.next
        node_before_first.next = node_after_second
        node_after_second.prev = node_before_first
        node_first.prev = None
        node_first.next = None
        node_second.prev = None
        node_second.next = None
        first_node_list = self.position_map[node_first.value]
        second_node_list = self.position_map[node_second.value]
        if node_first in first_node_list:
            print("Word: " + self.word + "delete node failed, first node doesn't exist in the list, first node value: " + value_first)
            return None
        if node_second in second_node_list:
            print("Word: " + self.word + "delete node failed, second node doesn't exist in the list, second node value: " + value_second)
            return None
        first_node_list.remove(node_first)
        second_node_list.remove(node_second)
        return node_before_first

    # Return the newly inserted node
    def insert(self, value, insert_position):
        if insert_position is None:
            print("Word: " + self.word + "insert node failed, insert position is None, insert value: " + value)
        if insert_position not in self.position_map:
            print("Word: " + self.word + "insert node failed, insert position doesn't exist in position map, insert value: " + value)
            return 
        new_node = Node(value, None, None)
        next_node = insert_position.next
        insert_position.next = new_node
        new_node.prev = insert_position
        new_node.next = next_node
        next_node.prev = new_node
        if value not in self.position_map:
            self.position_map[value] = []
        self.position_map[value].append(new_node)
        return new_node
