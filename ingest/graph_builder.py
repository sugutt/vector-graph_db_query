import os
import ast
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count  # <-- Import multiprocessing
import functools  # <-- Import functools

# --- AST Visitor (Modified to collect operations) ---
class AstDataCollectorVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to extract code structure and collects operations
    to be performed on the graph later. Does NOT interact with DB directly.
    """
    def __init__(self, file_path: str, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.current_class = None
        self.current_function = None
        self.current_scope_identifier = file_path
        self.operations = []  # List to store ('operation_name', *args) tuples

    # --- Scope Helpers (same as before) ---
    def _get_current_scope_id(self) -> str:
        if self.current_function:
            scope = f"{self.file_path}::{self.current_function}"
            if self.current_class:
                scope = f"{self.file_path}::{self.current_class}::{self.current_function}"
            return scope
        elif self.current_class:
            return f"{self.file_path}::{self.current_class}"
        else:
            return self.file_path

    # --- Visitor Methods (Modified to append operations) ---
    def visit_ClassDef(self, node):
        class_name = node.name
        self.operations.append(('add_class', self.file_path, class_name, node.lineno))

        for base in node.bases:
            if isinstance(base, ast.Name):
                base_name = base.id
                self.operations.append(('add_inheritance', self.file_path, class_name, base_name))

        original_class = self.current_class
        original_scope = self.current_scope_identifier
        self.current_class = class_name
        self.current_scope_identifier = self._get_current_scope_id()
        self.generic_visit(node)
        self.current_class = original_class
        self.current_scope_identifier = original_scope

    def visit_FunctionDef(self, node):
        func_name = node.name
        parent_name = self.current_class
        self.operations.append(('add_function', self.file_path, func_name, node.lineno, parent_name))

        original_function = self.current_function
        original_scope = self.current_scope_identifier
        self.current_function = func_name
        self.current_scope_identifier = self._get_current_scope_id()

        for arg in node.args.args:
            arg_name = arg.arg
            self.operations.append(('add_variable_definition',
                                     self.file_path, self.current_scope_identifier, arg_name, node.lineno, "argument"))

        self.generic_visit(node)
        self.current_function = original_function
        self.current_scope_identifier = original_scope

    def visit_Call(self, node):
        called_name = None
        if isinstance(node.func, ast.Name):
            called_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr

        if called_name and self.current_scope_identifier:
            self.operations.append(('add_call',
                                     self.file_path, self.current_scope_identifier, called_name, node.lineno))
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            alias_name = alias.asname if alias.asname else module_name
            is_internal = self._is_internal_module(module_name)
            module_type = "internal" if is_internal else "external"
            self.operations.append(('add_module', module_name, module_type))  # Ensure module exists
            self.operations.append(('add_import', self.file_path, module_name, alias_name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module_name = node.module if node.module else "."
        is_internal = self._is_internal_module(module_name)
        module_type = "internal" if is_internal else "external"
        self.operations.append(('add_module', module_name, module_type))  # Ensure module exists

        for alias in node.names:
            imported_name = alias.name
            alias_name = alias.asname if alias.asname else imported_name
            self.operations.append(('add_import_from',
                                     self.file_path, module_name, imported_name, alias_name, node.lineno))
        self.generic_visit(node)

    def _is_internal_module(self, module_name):
        if not module_name:
            return False
        parts = module_name.split('.')
        potential_path = self.project_root.joinpath(*parts)
        return potential_path.is_dir() or potential_path.with_suffix('.py').is_file()

    def visit_Assign(self, node):
        defined_vars = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                defined_vars.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        defined_vars.append(elt.id)

        for var_name in defined_vars:
            self.operations.append(('add_variable_definition',
                                     self.file_path, self.current_scope_identifier, var_name, node.lineno, "assignment"))
        self.generic_visit(node)

    def visit_Name(self, node):
        var_name = node.id
        if isinstance(node.ctx, ast.Load):
            self.operations.append(('add_variable_usage',
                                     self.file_path, self.current_scope_identifier, var_name, node.lineno, "read"))
        self.generic_visit(node)

    def visit_Return(self, node):
        if node.value:
            value_repr = ast.dump(node.value)
            self.operations.append(('add_return_value',
                                     self.file_path, self.current_scope_identifier, node.lineno, value_repr))
        self.generic_visit(node)


# --- Worker Function for AST Parsing ---
def _parse_python_file(file_path_tuple: tuple[Path, Path]) -> list:
    """
    Worker function to parse a single Python file and return graph operations.
    Takes a tuple (file_path, project_root).
    """
    file_path_obj, project_root = file_path_tuple
    file_path = str(file_path_obj)
    operations = [('add_file', file_path)]  # Start with adding the file itself
    try:
        with open(file_path_obj, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=file_path)
        # Use the data collector visitor
        visitor = AstDataCollectorVisitor(file_path, project_root)
        visitor.visit(tree)
        operations.extend(visitor.operations)  # Add operations found by visitor
    except SyntaxError as e:
        print(f"⚠️ Worker skipping {file_path} due to syntax error: {e}")
    except Exception as e:
        print(f"⚠️ Worker error processing {file_path} with AST: {e}")
    return operations


# --- CodeGraphBuilder Class (Modified build_graph_from_files) ---
class CodeGraphBuilder:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.project_root = None  # Will be set in build_graph_from_files

    def close(self):
        self._driver.close()

    def clear_graph(self):
        print("🗑️ Clearing existing graph data...")
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Graph cleared.")

    def build_graph_from_files(self, file_paths, project_root_path: Path):
        """
        Builds/Updates the graph by parsing Python files in parallel using AST
        and then executing the collected operations sequentially.
        """
        if not file_paths:
            print("📊 No files provided for graph building.")
            return

        print(f"\n🏗️ Building/Updating graph for {len(file_paths)} files...")
        self.project_root = project_root_path

        # Separate Python files for parallel AST parsing
        python_files = []
        non_python_files = []
        for p in file_paths:
            path_obj = Path(p) if not isinstance(p, Path) else p
            if path_obj.suffix == '.py':
                python_files.append(path_obj)
            else:
                non_python_files.append(path_obj)

        all_operations = []

        # Process non-Python files sequentially (just add file node)
        for file_path_obj in non_python_files:
            file_path = str(file_path_obj)
            print(f"\nProcessing non-Python file: {file_path}")
            all_operations.append(('add_file', file_path))
            print(f"  Skipping AST analysis for non-Python file: {file_path}")

        # Process Python files in parallel for AST parsing
        num_py_files = len(python_files)
        if num_py_files > 0:
            num_workers = max(1, cpu_count() - 1)
            print(f"🐍 Parsing {num_py_files} Python files using {num_workers} worker processes...")

            # Create tuples of (file_path, project_root) for the worker
            worker_args = [(py_file, self.project_root) for py_file in python_files]

            with Pool(processes=num_workers) as pool:
                results_iterator = pool.imap_unordered(_parse_python_file, worker_args)
                for file_ops in tqdm(results_iterator, total=num_py_files, desc="🐍 Parsing Python files"):
                    all_operations.extend(file_ops)

        # --- Execute all collected operations sequentially ---
        print(f"\n💾 Executing {len(all_operations)} graph operations...")
        op_counts = {}
        for operation in tqdm(all_operations, desc="💾 Applying graph operations"):
            op_name = operation[0]
            op_args = operation[1:]
            op_counts[op_name] = op_counts.get(op_name, 0) + 1
            try:
                # Get the corresponding method from self and call it
                method = getattr(self, op_name)
                method(*op_args)
            except AttributeError:
                print(f"  ❌ Unknown graph operation: {op_name}")
            except Exception as e:
                print(f"  ❌ Error executing operation {operation}: {e}")

        print("\n📊 Operation Counts:")
        for op_name, count in op_counts.items():
            print(f"  - {op_name}: {count}")

        print("\n✅ Graph building/update complete.")

    # --- Methods to add graph elements ---

    def add_file(self, file_path):
        with self._driver.session() as session:
            session.run("MERGE (f:File {path: $path})", path=file_path)

    def add_class(self, file_path, class_name, lineno):
        with self._driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $file_path})
                MERGE (c:Class {name: $class_name, file_path: $file_path})
                ON CREATE SET c.line = $lineno
                MERGE (f)-[:CONTAINS]->(c)
                """,
                file_path=file_path, class_name=class_name, lineno=lineno
            )

    def add_function(self, file_path, func_name, lineno, class_name=None):
        with self._driver.session() as session:
            scope_id = f"{file_path}::{func_name}"
            if class_name:
                scope_id = f"{file_path}::{class_name}::{func_name}"

            session.run(
                """
                MERGE (fn:Function {name: $func_name, file_path: $file_path, scope_id: $scope_id})
                ON CREATE SET fn.line = $lineno
                """,
                file_path=file_path, func_name=func_name, lineno=lineno, scope_id=scope_id
            )
            if class_name:
                session.run(
                    """
                    MATCH (c:Class {name: $class_name, file_path: $file_path})
                    MATCH (fn:Function {scope_id: $scope_id})
                    MERGE (c)-[:HAS_METHOD]->(fn)
                    """,
                    file_path=file_path, class_name=class_name, scope_id=scope_id
                )
            else:
                session.run(
                    """
                    MATCH (f:File {path: $file_path})
                    MATCH (fn:Function {scope_id: $scope_id})
                    MERGE (f)-[:CONTAINS]->(fn)
                    """,
                    file_path=file_path, scope_id=scope_id
                )

    def add_call(self, caller_file, caller_scope_id, called_function_name, lineno):
        with self._driver.session() as session:
            caller_match = """
                MATCH (caller) WHERE caller.scope_id = $caller_scope_id OR caller.path = $caller_scope_id
            """
            query = caller_match + """
                MERGE (called:Function {name: $called_func})
                MERGE (caller)-[r:CALLS]->(called)
                ON CREATE SET r.lines = [$lineno]
                ON MATCH SET r.lines = CASE WHEN NOT $lineno IN r.lines THEN r.lines + $lineno ELSE r.lines END
            """
            session.run(query,
                        caller_scope_id=caller_scope_id,
                        called_func=called_function_name,
                        lineno=lineno
                        )

    def add_inheritance(self, file_path, class_name, base_class_name):
        with self._driver.session() as session:
            session.run(
                """
                MATCH (c:Class {name: $class_name, file_path: $file_path})
                MERGE (base:Class {name: $base_class_name})
                MERGE (c)-[:INHERITS_FROM]->(base)
                """,
                file_path=file_path, class_name=class_name, base_class_name=base_class_name
            )

    def add_module(self, module_name, module_type="unknown"):
        with self._driver.session() as session:
            session.run(
                """
                MERGE (m:Module {name: $module_name})
                ON CREATE SET m.type = $module_type
                ON MATCH SET m.type = $module_type
                """,
                module_name=module_name, module_type=module_type
            )

    def add_import(self, file_path, module_name, alias_name, lineno):
        with self._driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $file_path})
                MATCH (m:Module {name: $module_name})
                MERGE (f)-[r:IMPORTS]->(m)
                ON CREATE SET r.alias = $alias, r.lines = [$lineno]
                ON MATCH SET r.alias = $alias, r.lines = CASE WHEN NOT $lineno IN r.lines THEN r.lines + $lineno ELSE r.lines END
                """,
                file_path=file_path, module_name=module_name, alias=alias_name, lineno=lineno
            )

    def add_import_from(self, file_path, module_name, imported_name, alias_name, lineno):
        with self._driver.session() as session:
            session.run(
                """
                MATCH (f:File {path: $file_path})
                MATCH (m:Module {name: $module_name})
                MERGE (f)-[r:IMPORTS_FROM {imported_name: $imported_name}]->(m)
                ON CREATE SET r.alias = $alias, r.lines = [$lineno]
                ON MATCH SET r.alias = $alias, r.lines = CASE WHEN NOT $lineno IN r.lines THEN r.lines + $lineno ELSE r.lines END
                """,
                file_path=file_path, module_name=module_name, imported_name=imported_name, alias=alias_name, lineno=lineno
            )

    def add_variable_definition(self, file_path, scope_id, var_name, lineno, definition_type):
        with self._driver.session() as session:
            scope_match = """
                MATCH (scope) WHERE scope.scope_id = $scope_id OR scope.path = $scope_id
            """
            query = scope_match + """
                MERGE (v:Variable {name: $var_name, scope_id: $scope_id})
                ON CREATE SET v.defined_at = $lineno, v.definition_type = $def_type
                MERGE (scope)-[:DEFINES]->(v)
            """
            session.run(query,
                        scope_id=scope_id,
                        var_name=var_name,
                        lineno=lineno,
                        def_type=definition_type
                        )

    def add_variable_usage(self, file_path, scope_id, var_name, lineno, usage_type):
        with self._driver.session() as session:
            scope_match = """
                MATCH (scope) WHERE scope.scope_id = $scope_id OR scope.path = $scope_id
            """
            variable_match = """
                MATCH (v:Variable {name: $var_name, scope_id: $scope_id})
            """
            rel_type = "READS" if usage_type == "read" else "WRITES"
            query = scope_match + variable_match + f"""
                MERGE (scope)-[r:{rel_type}]->(v)
                ON CREATE SET r.lines = [$lineno]
                ON MATCH SET r.lines = CASE WHEN NOT $lineno IN r.lines THEN r.lines + $lineno ELSE r.lines END
            """
            try:
                session.run(query, scope_id=scope_id, var_name=var_name, lineno=lineno)
            except Exception as e:
                print(f"  WARN: Could not link usage of '{var_name}' in scope '{Path(scope_id).name}' at line {lineno}. Variable definition might be out of scope or analysis is limited. Error: {e}")

    def add_return_value(self, file_path, scope_id, lineno, value_repr):
        with self._driver.session() as session:
            session.run(
                """
                MATCH (fn:Function {scope_id: $scope_id})
                MERGE (fn)-[r:RETURNS]->(fn)
                ON CREATE SET r.lines = [$lineno], r.value_repr = [$value_repr]
                ON MATCH SET r.lines = CASE WHEN NOT $lineno IN r.lines THEN r.lines + $lineno ELSE r.lines END,
                              r.value_repr = CASE WHEN NOT $lineno IN r.lines THEN r.value_repr + $value_repr ELSE r.value_repr END
                """,
                scope_id=scope_id, lineno=lineno, value_repr=value_repr
            )
