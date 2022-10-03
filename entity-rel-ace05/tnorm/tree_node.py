import torch

class TreeNode:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        arg_str = "(" + ",".join([str(c) for c in self.children if not type(c) == torch.Tensor]) + ")" if len(self.children) > 0 else ""
        return self.name + arg_str

    def as_bool(self):
        return self

    def And(self, other):
        if other is None:
            return self
        return And(other, self)

    def Or(self, other):
        if other is None:
            return self
        return Or(other, self)

    def __eq__(self, obj):
        return (type(obj) == type(self) and
                self.name == obj.name and self.children == obj.children)

class TreeNodeVisitor:

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        print(self)
        raise NotImplementedError


class BinaryOp(TreeNode):
    def __init__(self, name, left, right):
        self.left = left
        self.right = right
        super().__init__(name, [left, right])


class And(BinaryOp):
    def __init__(self, left, right):
        super().__init__("And", left, right)


class Or(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Or", left, right)


class Implication(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Implication", left, right)


class UnaryOp(TreeNode):
    def __init__(self, name, operand):
        self.operand = operand
        super().__init__(name, [operand])


class Not(UnaryOp):
    def __init__(self, operand):
        super().__init__("Not", operand)


class IsEq(BinaryOp):
    def __init__(self, left, right):
        super().__init__('Eq', left, right)

class TNormTreeNodeVisitor(TreeNodeVisitor):

    def visit_IsEq(self, node):
        if isinstance(node.left, torch.Tensor) and isinstance(node.right, int):
            return node.left[:, [node.right]]

class ProductTNormVisitor(TNormTreeNodeVisitor):

    def visit_And(self, node):
        lv = self.visit(node.left)
        rv = self.visit(node.right)
        return lv * rv

    def visit_Or(self, node):
        lv = self.visit(node.left)
        rv = self.visit(node.right)
        return lv + rv - lv * rv

    def visit_Implication(self, node):
        lv = self.visit(node.left)
        rv = self.visit(node.right)
        return 1 - torch.relu(1 - rv / (lv + 1e-5))
