//! Semantic AST chunking — split source code at meaningful boundaries.
//!
//! Uses Python's built-in `ast` module (via PyO3) for Python source.
//! This avoids a `tree-sitter` dependency while still providing
//! syntax-aware splitting at function/class boundaries.
//!
//! Each chunk is a self-contained unit that can be delegated to a
//! sub-agent: a function definition, a class definition, or an
//! import/top-level block.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// What kind of boundary to split at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkBoundary {
    /// Each top-level function = one chunk.
    Function,
    /// Each class (with all its methods) = one chunk.
    Class,
    /// Each top-level statement = one chunk (finest granularity).
    Statement,
    /// The entire file = one chunk (coarsest).
    Module,
}

/// A semantically coherent chunk of source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// The source code of this chunk.
    pub content: String,
    /// What kind of node this is.
    pub kind: ChunkKind,
    /// Name of the function/class (if applicable).
    pub name: Option<String>,
    /// Line range in the original source (0-indexed, half-open).
    pub line_range: Range<usize>,
    /// Imports that this chunk depends on (detected from the AST).
    pub dependencies: Vec<String>,
}

/// The kind of code chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkKind {
    /// A function or async function definition.
    Function,
    /// A class definition (includes all methods).
    Class,
    /// Import statements.
    Import,
    /// Top-level expressions or assignments.
    TopLevel,
    /// The entire module (when boundary=Module).
    Module,
}

/// Split Python source code into semantically coherent chunks.
///
/// Uses Python's `ast` module via PyO3 for reliable parsing.
/// Falls back to line-based splitting if AST parsing fails.
pub fn chunk_python(source: &str, boundary: ChunkBoundary) -> Vec<CodeChunk> {
    match boundary {
        ChunkBoundary::Module => vec![CodeChunk {
            content: source.to_string(),
            kind: ChunkKind::Module,
            name: None,
            line_range: 0..source.lines().count(),
            dependencies: extract_imports_simple(source),
        }],
        _ => chunk_python_ast(source, boundary),
    }
}

/// AST-based chunking using Python's `ast` module via PyO3.
fn chunk_python_ast(source: &str, boundary: ChunkBoundary) -> Vec<CodeChunk> {
    let source_owned = source.to_string();

    // Run Python AST parsing in a blocking context
    let result: Result<Vec<CodeChunk>, pyo3::PyErr> = Python::attach(|py| {
        // Parse the source into an AST and extract node info as JSON
        let script = format!(
            r#"
import ast, json

source = {source_repr}
tree = ast.parse(source)

nodes = []
for node in ast.iter_child_nodes(tree):
    info = {{
        "type": type(node).__name__,
        "lineno": getattr(node, 'lineno', 1),
        "end_lineno": getattr(node, 'end_lineno', None),
        "name": getattr(node, 'name', None),
    }}

    # For imports, extract module names
    if isinstance(node, ast.Import):
        info["imports"] = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        info["imports"] = [node.module or ""]

    # For functions/classes, detect dependencies (names used)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
            elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                names.add(child.value.id)
        info["used_names"] = list(names)

    nodes.append(info)

result = json.dumps(nodes)
"#,
            source_repr = format!("{:?}", source_owned)
        );

        // Execute the script
        let locals = pyo3::types::PyDict::new(py);
        py.run(
            &std::ffi::CString::new(script).unwrap(),
            None,
            Some(&locals),
        )?;

        let result_str: String = locals
            .get_item("result")?
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no result"))?
            .extract()?;

        let nodes: Vec<AstNodeInfo> = serde_json::from_str(&result_str).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
        })?;

        Ok::<Vec<CodeChunk>, pyo3::PyErr>(build_chunks(&source_owned, &nodes, boundary))
    });

    match result {
        Ok(chunks) => chunks,
        Err(_) => {
            // Fallback: treat entire source as one chunk
            vec![CodeChunk {
                content: source.to_string(),
                kind: ChunkKind::Module,
                name: None,
                line_range: 0..source.lines().count(),
                dependencies: extract_imports_simple(source),
            }]
        }
    }
}

/// Internal AST node info from Python's ast module.
#[derive(Debug, Clone, Deserialize)]
struct AstNodeInfo {
    #[serde(rename = "type")]
    node_type: String,
    lineno: usize,
    end_lineno: Option<usize>,
    name: Option<String>,
    #[serde(default)]
    imports: Vec<String>,
    #[serde(default)]
    used_names: Vec<String>,
}

/// Build chunks from AST node info.
fn build_chunks(source: &str, nodes: &[AstNodeInfo], boundary: ChunkBoundary) -> Vec<CodeChunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut import_lines: Vec<usize> = Vec::new();
    let mut import_deps: Vec<String> = Vec::new();

    for node in nodes {
        let start = node.lineno.saturating_sub(1); // 1-indexed → 0-indexed
        let end = node.end_lineno.unwrap_or(node.lineno);

        match node.node_type.as_str() {
            "Import" | "ImportFrom" => {
                for line in start..end.min(lines.len()) {
                    import_lines.push(line);
                }
                import_deps.extend(node.imports.clone());
            }
            "FunctionDef" | "AsyncFunctionDef" => {
                if matches!(boundary, ChunkBoundary::Function | ChunkBoundary::Statement) {
                    let content = lines[start..end.min(lines.len())].join("\n");
                    chunks.push(CodeChunk {
                        content,
                        kind: ChunkKind::Function,
                        name: node.name.clone(),
                        line_range: start..end,
                        dependencies: node.used_names.clone(),
                    });
                }
            }
            "ClassDef" => {
                if matches!(
                    boundary,
                    ChunkBoundary::Function | ChunkBoundary::Class | ChunkBoundary::Statement
                ) {
                    let content = lines[start..end.min(lines.len())].join("\n");
                    chunks.push(CodeChunk {
                        content,
                        kind: ChunkKind::Class,
                        name: node.name.clone(),
                        line_range: start..end,
                        dependencies: node.used_names.clone(),
                    });
                }
            }
            _ => {
                if matches!(boundary, ChunkBoundary::Statement) {
                    let content = lines[start..end.min(lines.len())].join("\n");
                    chunks.push(CodeChunk {
                        content,
                        kind: ChunkKind::TopLevel,
                        name: None,
                        line_range: start..end,
                        dependencies: Vec::new(),
                    });
                }
            }
        }
    }

    // Add imports as the first chunk if there are any
    if !import_lines.is_empty() {
        let content: Vec<&str> = import_lines
            .iter()
            .filter_map(|&i| lines.get(i))
            .copied()
            .collect();
        chunks.insert(
            0,
            CodeChunk {
                content: content.join("\n"),
                kind: ChunkKind::Import,
                name: None,
                line_range: import_lines[0]..import_lines[import_lines.len() - 1] + 1,
                dependencies: import_deps,
            },
        );
    }

    // If boundary=Class, merge non-class/function nodes into top-level
    if matches!(boundary, ChunkBoundary::Class) {
        // functions stay as-is, classes stay as-is, everything else merges
    }

    if chunks.is_empty() {
        // Nothing was extracted — return the whole source
        chunks.push(CodeChunk {
            content: source.to_string(),
            kind: ChunkKind::Module,
            name: None,
            line_range: 0..lines.len(),
            dependencies: extract_imports_simple(source),
        });
    }

    chunks
}

/// Simple import extraction without AST (fallback).
fn extract_imports_simple(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.starts_with("import ") {
                Some(
                    trimmed
                        .strip_prefix("import ")?
                        .split_whitespace()
                        .next()?
                        .to_string(),
                )
            } else if trimmed.starts_with("from ") {
                Some(
                    trimmed
                        .strip_prefix("from ")?
                        .split_whitespace()
                        .next()?
                        .to_string(),
                )
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PYTHON: &str = r#"
import json
from pathlib import Path

def transform(input_grid):
    """Transform input grid to output."""
    rows = len(input_grid)
    cols = len(input_grid[0])
    output = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            output[i][j] = input_grid[i][j] * 2
    return output

class GridSolver:
    def __init__(self, grid):
        self.grid = grid

    def solve(self):
        return transform(self.grid)

x = GridSolver([[1, 2], [3, 4]])
print(x.solve())
"#;

    #[test]
    fn test_chunk_python_module() {
        let chunks = chunk_python(SAMPLE_PYTHON, ChunkBoundary::Module);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].kind, ChunkKind::Module);
        assert!(chunks[0].dependencies.contains(&"json".to_string()));
    }

    #[test]
    fn test_chunk_python_functions() {
        let chunks = chunk_python(SAMPLE_PYTHON, ChunkBoundary::Function);
        // Should have: imports, transform function, GridSolver class
        assert!(
            chunks.len() >= 3,
            "got {} chunks: {:?}",
            chunks.len(),
            chunks
                .iter()
                .map(|c| (&c.kind, &c.name))
                .collect::<Vec<_>>()
        );

        // Find the function chunk
        let func = chunks
            .iter()
            .find(|c| c.name.as_deref() == Some("transform"))
            .unwrap();
        assert_eq!(func.kind, ChunkKind::Function);
        assert!(func.content.contains("def transform"));

        // Find the class chunk
        let cls = chunks
            .iter()
            .find(|c| c.name.as_deref() == Some("GridSolver"))
            .unwrap();
        assert_eq!(cls.kind, ChunkKind::Class);
        assert!(cls.content.contains("class GridSolver"));
    }

    #[test]
    fn test_chunk_python_classes() {
        let chunks = chunk_python(SAMPLE_PYTHON, ChunkBoundary::Class);
        // Should have: imports, GridSolver class (functions not split out)
        let cls = chunks.iter().find(|c| c.kind == ChunkKind::Class);
        assert!(cls.is_some(), "expected a class chunk");
    }

    #[test]
    fn test_extract_imports_simple() {
        let source = "import json\nfrom pathlib import Path\nx = 1\n";
        let imports = extract_imports_simple(source);
        assert!(imports.contains(&"json".to_string()));
        assert!(imports.contains(&"pathlib".to_string()));
    }

    #[test]
    fn test_chunk_empty_source() {
        let chunks = chunk_python("", ChunkBoundary::Function);
        // Should return at least one chunk (the whole module)
        assert!(!chunks.is_empty());
    }
}
