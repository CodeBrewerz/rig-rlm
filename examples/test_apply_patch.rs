// Quick standalone test of apply_patch module
use std::fs;
use std::path::Path;

fn main() {
    // Setup: create a temp dir with a test file
    let dir = std::env::temp_dir().join("rig_rlm_patch_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let test_file = dir.join("hello.py");
    fs::write(&test_file, "def hello():\n    print(\"hello\")\n    pass\n").unwrap();

    println!("=== Test 1: Parse + Apply Update Patch ===");
    let patch = r#"
--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,4 @@
 def hello():
-    print("hello")
+    print("hello world")
+    return True
     pass
"#;

    // Parse
    let action = rig_rlm::apply_patch::parse_patch(patch).unwrap();
    println!(
        "  Parsed: {} file(s) — {}",
        action.file_count(),
        action.summary()
    );

    // Apply
    let results = rig_rlm::apply_patch::apply_patch(&action, &dir).unwrap();
    for (path, desc) in &results {
        println!("  Applied: {} — {}", path.display(), desc);
    }

    // Verify
    let content = fs::read_to_string(&test_file).unwrap();
    assert!(
        content.contains("hello world"),
        "should contain 'hello world'"
    );
    assert!(
        content.contains("return True"),
        "should contain 'return True'"
    );
    assert!(
        !content.contains("print(\"hello\")"),
        "should NOT contain old line"
    );
    println!("  ✅ Content verified!");
    println!("  Content:\n{}", content);

    println!("\n=== Test 2: Add New File via Patch ===");
    let add_patch = r#"
--- /dev/null
+++ b/new_module.py
@@ -0,0 +1,3 @@
+def new_func():
+    return 42
+
"#;
    let action2 = rig_rlm::apply_patch::parse_patch(add_patch).unwrap();
    println!(
        "  Parsed: {} file(s) — {}",
        action2.file_count(),
        action2.summary()
    );
    let results2 = rig_rlm::apply_patch::apply_patch(&action2, &dir).unwrap();
    for (path, desc) in &results2 {
        println!("  Applied: {} — {}", path.display(), desc);
    }
    let new_file = dir.join("new_module.py");
    assert!(new_file.exists(), "new file should exist");
    let content2 = fs::read_to_string(&new_file).unwrap();
    assert!(content2.contains("return 42"), "should contain 'return 42'");
    println!("  ✅ New file created and verified!");

    println!("\n=== Test 3: Delete File via Patch ===");
    let del_patch = r#"
--- a/new_module.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def new_func():
-    return 42
-
"#;
    let action3 = rig_rlm::apply_patch::parse_patch(del_patch).unwrap();
    println!(
        "  Parsed: {} file(s) — {}",
        action3.file_count(),
        action3.summary()
    );
    let results3 = rig_rlm::apply_patch::apply_patch(&action3, &dir).unwrap();
    for (path, desc) in &results3 {
        println!("  Applied: {} — {}", path.display(), desc);
    }
    assert!(!new_file.exists(), "file should be deleted");
    println!("  ✅ File deleted!");

    println!("\n=== Test 4: ExecPolicy evaluates patch ===");
    let policy = rig_rlm::exec_policy::ExecPolicy::standard();

    // Safe patch
    let safe_eval = policy.evaluate(patch);
    println!(
        "  Safe patch: decision={}, allowed={}",
        safe_eval.decision,
        safe_eval.is_allowed()
    );
    assert!(safe_eval.is_allowed(), "safe patch should be allowed");

    // Dangerous patch
    let dangerous = "rm -rf /\nsudo chmod 777 /etc/shadow\n";
    let danger_eval = policy.evaluate(dangerous);
    println!(
        "  Dangerous patch: decision={}, denied={}",
        danger_eval.decision,
        danger_eval.is_denied()
    );
    println!("  Reason: {}", danger_eval.reason());
    assert!(danger_eval.is_denied(), "dangerous patch should be denied");
    println!("  ✅ ExecPolicy correctly evaluates patches!");

    // Cleanup
    let _ = fs::remove_dir_all(&dir);

    println!("\n=== ALL APPLY-PATCH E2E TESTS PASSED ✅ ===");
}
