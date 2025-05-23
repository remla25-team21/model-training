import astroid
from pylint.checkers import BaseChecker

class HardcodedSeedChecker(BaseChecker):
    # Remove the __implements__ line - it's no longer needed
    
    name = 'hardcoded-seed-checker'
    msgs = {
        'C9001': (
            'Hardcoded random seed detected. Use configurable seed for reproducibility.',
            'hardcoded-seed',
            'Avoid hardcoded seeds like random.seed(42).'
        ),
    }

    def visit_call(self, node):
        """Check for hardcoded seed calls in random operations."""
        try:
            func_name = ''
            # Check if this is an attribute call (like numpy.random.seed)
            if hasattr(node.func, 'attrname'):
                func_name = node.func.attrname
            # Check if this is a direct function call (like seed)
            elif hasattr(node.func, 'name'):
                func_name = node.func.name
            
            # Flag any call to a function named 'seed'
            if func_name == 'seed':
                self.add_message('hardcoded-seed', node=node)
        except Exception:
            # Silently handle any unexpected node structures
            pass

class UnnecessaryIterationPandasChecker(BaseChecker):
    # Remove the __implements__ line - it's no longer needed
    
    name = 'unnecessary-iteration-pandas-checker'
    msgs = {
        'C9002': (
            'Unnecessary iteration over pandas object detected. Use vectorized operations instead.',
            'unnecessary-iteration-pandas',
            'Avoid explicit loops on pandas DataFrames or Series.'
        ),
    }

    def visit_for(self, node):
        """Check for inefficient pandas iterations like iterrows()."""
        try:
            # Check if we're iterating over a method call
            if hasattr(node.iter, 'func'):
                # Handle attribute method calls (df.iterrows())
                if hasattr(node.iter.func, 'attrname') and node.iter.func.attrname == 'iterrows':
                    self.add_message('unnecessary-iteration-pandas', node=node)
                # Handle direct function calls (iterrows())
                elif hasattr(node.iter.func, 'name') and node.iter.func.name == 'iterrows':
                    self.add_message('unnecessary-iteration-pandas', node=node)
        except Exception:
            # Handle any unexpected node structures gracefully
            pass

class NanEquivalenceChecker(BaseChecker):
    # Remove the __implements__ line - it's no longer needed
    
    name = 'nan-equivalence'
    msgs = {
        'W0001': (
            'Use .isna() for NaN comparison.',
            'nan-equivalence',
            'Developers need to be careful when using the NaN comparison',
        ),
    }

    def visit_compare(self, node):
        """Check for direct NaN comparisons that should use .isna() instead."""
        try:
            # Check if we have any comparison operations
            if len(node.ops) > 0 and node.ops[0][0] == '==':
                # Look for patterns like df == np.nan
                if isinstance(node.left, astroid.Name) and node.left.name == 'df':
                    # Check the right side of the comparison
                    if (len(node.ops[0]) >= 2 and 
                        isinstance(node.ops[0][1], astroid.Attribute) and 
                        node.ops[0][1].attrname == 'nan' and 
                        isinstance(node.ops[0][1].expr, astroid.Name) and 
                        node.ops[0][1].expr.name == 'np'):
                        self.add_message('nan-equivalence', node=node)
        except Exception:
            # Handle any unexpected node structures gracefully
            pass

def register(linter):
    """Register all custom checkers with the pylint linter."""
    linter.register_checker(HardcodedSeedChecker(linter))
    linter.register_checker(UnnecessaryIterationPandasChecker(linter))
    linter.register_checker(NanEquivalenceChecker(linter))