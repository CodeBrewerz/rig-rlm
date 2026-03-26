//! Real-world knowledge graph test: RDF-style triples + multi-hop reasoning.
//!
//! Stores facts as (subject, predicate, object) triples and tests:
//!   1. Direct lookups: "What company does Alice work for?"
//!   2. Multi-hop (2-hop): "Who is the CEO of the company Alice works for?"
//!   3. Multi-hop (3-hop): "What city is the HQ of the company Alice works for in?"
//!   4. Reverse lookups: "Who works at Google?"
//!   5. N-Quad style: context-scoped facts (graph-aware)
//!
//! Run: cargo test -p rig-rlm --lib nuggets::test_knowledge_graph -- --nocapture

#[cfg(test)]
mod tests {
    use crate::nuggets::memory::{Nugget, NuggetOpts};

    fn make_nugget(name: &str) -> Nugget {
        Nugget::new(NuggetOpts {
            name: name.into(),
            d: 2048,
            banks: 4,
            auto_save: false,
            ..Default::default()
        })
    }

    // ═══════════════════════════════════════════════════════════════
    // §1  Triple Store — direct lookups
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn triple_store_direct_lookups() {
        println!("\n{}", "═".repeat(70));
        println!("  §1  TRIPLE STORE — Direct Lookups");
        println!("{}\n", "═".repeat(70));

        let mut n = make_nugget("kg_triples");

        // Store triples as "subject:predicate" → "object"
        let triples = vec![
            // People
            ("Alice:worksAt", "Google"),
            ("Alice:role", "Senior Engineer"),
            ("Alice:livesIn", "San Francisco"),
            ("Alice:age", "32"),
            ("Alice:knows", "Bob"),
            ("Alice:almaMater", "MIT"),

            ("Bob:worksAt", "Meta"),
            ("Bob:role", "VP Engineering"),
            ("Bob:livesIn", "Menlo Park"),
            ("Bob:age", "45"),
            ("Bob:knows", "Charlie"),
            ("Bob:almaMater", "Stanford"),

            ("Charlie:worksAt", "Apple"),
            ("Charlie:role", "Staff Designer"),
            ("Charlie:livesIn", "Cupertino"),
            ("Charlie:knows", "Alice"),
            ("Charlie:almaMater", "RISD"),

            ("David:worksAt", "Microsoft"),
            ("David:role", "Principal PM"),
            ("David:livesIn", "Seattle"),
            ("David:almaMater", "Carnegie Mellon"),

            ("Eve:worksAt", "Amazon"),
            ("Eve:role", "Distinguished Engineer"),
            ("Eve:livesIn", "Seattle"),
            ("Eve:almaMater", "Berkeley"),

            // Companies
            ("Google:ceo", "Sundar Pichai"),
            ("Google:hq", "Mountain View"),
            ("Google:founded", "1998"),
            ("Google:industry", "Technology"),
            ("Google:employees", "180000"),

            ("Meta:ceo", "Mark Zuckerberg"),
            ("Meta:hq", "Menlo Park"),
            ("Meta:founded", "2004"),
            ("Meta:industry", "Social Media"),

            ("Apple:ceo", "Tim Cook"),
            ("Apple:hq", "Cupertino"),
            ("Apple:founded", "1976"),
            ("Apple:industry", "Consumer Electronics"),

            ("Microsoft:ceo", "Satya Nadella"),
            ("Microsoft:hq", "Redmond"),
            ("Microsoft:founded", "1975"),

            ("Amazon:ceo", "Andy Jassy"),
            ("Amazon:hq", "Seattle"),
            ("Amazon:founded", "1994"),

            // Cities
            ("Mountain View:state", "California"),
            ("Mountain View:country", "USA"),
            ("Menlo Park:state", "California"),
            ("Cupertino:state", "California"),
            ("Seattle:state", "Washington"),
            ("Redmond:state", "Washington"),
            ("San Francisco:state", "California"),
            ("San Francisco:population", "874000"),

            // Universities
            ("MIT:location", "Cambridge"),
            ("Stanford:location", "Stanford"),
            ("Berkeley:location", "Berkeley"),
            ("Carnegie Mellon:location", "Pittsburgh"),
            ("RISD:location", "Providence"),
        ];

        for (key, value) in &triples {
            n.remember(key, value);
        }

        println!("  Stored {} triples\n", triples.len());

        // Direct lookups
        let queries = vec![
            ("Alice:worksAt", "Google"),
            ("Bob:role", "VP Engineering"),
            ("Google:ceo", "Sundar Pichai"),
            ("Apple:hq", "Cupertino"),
            ("Seattle:state", "Washington"),
            ("MIT:location", "Cambridge"),
            ("Eve:almaMater", "Berkeley"),
            ("Meta:founded", "2004"),
        ];

        let mut correct = 0;
        println!("  {:<30} {:<25} {:<25} {}", "Query", "Expected", "Got", "✓/✗");
        println!("  {:-<95}", "");

        for (key, expected) in &queries {
            let result = n.recall(key, "kg");
            let got = result.answer.as_deref().unwrap_or("NOT FOUND");
            let ok = got == *expected;
            if ok { correct += 1; }
            println!("  {:<30} {:<25} {:<25} {}", key, expected, got, if ok { "✓" } else { "✗" });
        }

        println!("\n  Direct lookups: {}/{} correct\n", correct, queries.len());
        assert_eq!(correct, queries.len(), "All direct lookups should succeed");
    }

    // ═══════════════════════════════════════════════════════════════
    // §2  Multi-hop reasoning
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn multi_hop_reasoning() {
        println!("\n{}", "═".repeat(70));
        println!("  §2  MULTI-HOP REASONING");
        println!("{}\n", "═".repeat(70));

        let mut n = make_nugget("kg_multihop");

        // Store the knowledge graph
        let triples = vec![
            ("Alice:worksAt", "Google"),
            ("Bob:worksAt", "Meta"),
            ("Charlie:worksAt", "Apple"),
            ("David:worksAt", "Microsoft"),
            ("Eve:worksAt", "Amazon"),

            ("Google:ceo", "Sundar Pichai"),
            ("Meta:ceo", "Mark Zuckerberg"),
            ("Apple:ceo", "Tim Cook"),
            ("Microsoft:ceo", "Satya Nadella"),
            ("Amazon:ceo", "Andy Jassy"),

            ("Google:hq", "Mountain View"),
            ("Meta:hq", "Menlo Park"),
            ("Apple:hq", "Cupertino"),
            ("Microsoft:hq", "Redmond"),
            ("Amazon:hq", "Seattle"),

            ("Mountain View:state", "California"),
            ("Menlo Park:state", "California"),
            ("Cupertino:state", "California"),
            ("Redmond:state", "Washington"),
            ("Seattle:state", "Washington"),

            ("California:country", "USA"),
            ("Washington:country", "USA"),

            ("Alice:almaMater", "MIT"),
            ("Bob:almaMater", "Stanford"),
            ("MIT:location", "Cambridge"),
            ("Stanford:location", "Stanford"),
            ("Cambridge:state", "Massachusetts"),
        ];

        for (k, v) in &triples {
            n.remember(k, v);
        }

        println!("  Stored {} triples\n", triples.len());

        // ── 2-hop: "Who is the CEO of Alice's company?" ─────────
        println!("  ── 2-HOP QUERIES ──\n");

        let two_hop_queries = vec![
            ("Alice", "worksAt", "ceo", "Sundar Pichai", "Who is the CEO of Alice's company?"),
            ("Bob", "worksAt", "ceo", "Mark Zuckerberg", "Who is the CEO of Bob's company?"),
            ("Charlie", "worksAt", "hq", "Cupertino", "What city is Apple's HQ in?"),
            ("David", "worksAt", "hq", "Redmond", "Where is Microsoft's HQ?"),
            ("Eve", "worksAt", "ceo", "Andy Jassy", "Who is the CEO of Eve's company?"),
        ];

        let mut correct_2hop = 0;
        for (person, rel1, rel2, expected, question) in &two_hop_queries {
            // Hop 1: person:rel1 → company
            let hop1_key = format!("{person}:{rel1}");
            let hop1 = n.recall(&hop1_key, "kg");
            let company = hop1.answer.as_deref().unwrap_or("?");

            // Hop 2: company:rel2 → answer
            let hop2_key = format!("{company}:{rel2}");
            let hop2 = n.recall(&hop2_key, "kg");
            let answer = hop2.answer.as_deref().unwrap_or("?");

            let ok = answer == *expected;
            if ok { correct_2hop += 1; }

            println!("  Q: {question}");
            println!("    Hop 1: {hop1_key} → {company}");
            println!("    Hop 2: {hop2_key} → {answer}");
            println!("    Expected: {expected}  {}\n", if ok { "✓" } else { "✗" });
        }

        // ── 3-hop: "What state is the HQ of Alice's company in?" ─
        println!("  ── 3-HOP QUERIES ──\n");

        let three_hop_queries = vec![
            ("Alice", "worksAt", "hq", "state", "California",
             "What state is the HQ of Alice's company in?"),
            ("David", "worksAt", "hq", "state", "Washington",
             "What state is Microsoft's HQ in?"),
            ("Alice", "almaMater", "location", "state", "Massachusetts",
             "What state is Alice's alma mater in?"),
        ];

        let mut correct_3hop = 0;
        for (person, rel1, rel2, rel3, expected, question) in &three_hop_queries {
            let hop1 = n.recall(&format!("{person}:{rel1}"), "kg");
            let entity1 = hop1.answer.as_deref().unwrap_or("?");

            let hop2 = n.recall(&format!("{entity1}:{rel2}"), "kg");
            let entity2 = hop2.answer.as_deref().unwrap_or("?");

            let hop3 = n.recall(&format!("{entity2}:{rel3}"), "kg");
            let answer = hop3.answer.as_deref().unwrap_or("?");

            let ok = answer == *expected;
            if ok { correct_3hop += 1; }

            println!("  Q: {question}");
            println!("    Hop 1: {person}:{rel1} → {entity1}");
            println!("    Hop 2: {entity1}:{rel2} → {entity2}");
            println!("    Hop 3: {entity2}:{rel3} → {answer}");
            println!("    Expected: {expected}  {}\n", if ok { "✓" } else { "✗" });
        }

        println!("  2-hop: {correct_2hop}/{} correct", two_hop_queries.len());
        println!("  3-hop: {correct_3hop}/{} correct", three_hop_queries.len());

        assert_eq!(correct_2hop, two_hop_queries.len(), "All 2-hop queries should succeed");
        assert_eq!(correct_3hop, three_hop_queries.len(), "All 3-hop queries should succeed");
    }

    // ═══════════════════════════════════════════════════════════════
    // §3  N-Quad style — graph-scoped facts
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn nquad_graph_scoped_facts() {
        println!("\n{}", "═".repeat(70));
        println!("  §3  N-QUAD STYLE — Graph-Scoped Facts");
        println!("{}\n", "═".repeat(70));

        // Each graph is a separate nugget — like named graphs in RDF
        let mut prod = make_nugget("graph_production");
        let mut staging = make_nugget("graph_staging");
        let mut personal = make_nugget("graph_personal");

        // Production graph: current state of the world
        let prod_facts = vec![
            ("api:version", "v3.2.1"),
            ("api:status", "healthy"),
            ("db:primary", "pg-prod-01.us-east-1"),
            ("db:replicas", "3"),
            ("cache:provider", "Redis"),
            ("cache:hitRate", "94.2%"),
            ("deploy:lastDeploy", "2026-03-25T10:00:00Z"),
            ("deploy:method", "blue-green"),
        ];

        // Staging graph: different values for same keys
        let staging_facts = vec![
            ("api:version", "v3.3.0-rc1"),
            ("api:status", "testing"),
            ("db:primary", "pg-staging-01.us-east-1"),
            ("db:replicas", "1"),
            ("cache:provider", "Redis"),
            ("cache:hitRate", "87.1%"),
            ("deploy:lastDeploy", "2026-03-25T14:30:00Z"),
            ("deploy:method", "rolling"),
        ];

        // Personal knowledge graph
        let personal_facts = vec![
            ("project:current", "TurboQuant integration"),
            ("project:language", "Rust"),
            ("preference:editor", "VS Code"),
            ("preference:theme", "Dark"),
            ("todo:next", "benchmark release mode"),
        ];

        for (k, v) in &prod_facts { prod.remember(k, v); }
        for (k, v) in &staging_facts { staging.remember(k, v); }
        for (k, v) in &personal_facts { personal.remember(k, v); }

        println!("  Stored {} prod, {} staging, {} personal facts\n",
            prod_facts.len(), staging_facts.len(), personal_facts.len());

        // Same key, different graphs → different answers
        println!("  ── Same key, different graphs ──\n");
        println!("  {:<20} {:<20} {:<20} {:<20}", "Key", "Production", "Staging", "Personal");
        println!("  {:-<80}", "");

        let shared_keys = ["api:version", "api:status", "db:primary", "cache:hitRate"];
        for key in &shared_keys {
            let p = prod.recall(key, "nq").answer.as_deref().unwrap_or("-").to_string();
            let s = staging.recall(key, "nq").answer.as_deref().unwrap_or("-").to_string();
            let r = personal.recall(key, "nq").answer.as_deref().unwrap_or("-").to_string();
            println!("  {:<20} {:<20} {:<20} {:<20}", key, p, s, r);
        }

        // Verify isolation
        let prod_version = prod.recall("api:version", "nq").answer.unwrap();
        let staging_version = staging.recall("api:version", "nq").answer.unwrap();
        assert_eq!(prod_version, "v3.2.1");
        assert_eq!(staging_version, "v3.3.0-rc1");
        assert_ne!(prod_version, staging_version);

        // Personal graph is isolated — prod has no "project:" keys
        let personal_project = personal.recall("project:current", "nq").answer.unwrap();
        assert_eq!(personal_project, "TurboQuant integration");

        // Prod's best fuzzy match for "project:current" should NOT be the personal answer
        let prod_closest = prod.recall("project:current", "nq");
        let prod_answer = prod_closest.answer.as_deref().unwrap_or("");
        assert_ne!(prod_answer, "TurboQuant integration",
            "prod graph should not contain personal graph data");

        println!("\n  ✓ Graph isolation verified — same keys, different answers per graph");
    }

    // ═══════════════════════════════════════════════════════════════
    // §4  Larger knowledge graph with multi-hop
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn large_kg_multi_hop() {
        println!("\n{}", "═".repeat(70));
        println!("  §4  LARGE KG — 500 triples, multi-hop");
        println!("{}\n", "═".repeat(70));

        let mut n = make_nugget("kg_large");

        // Generate a synthetic but structured knowledge graph
        let cities = ["NYC", "LA", "Chicago", "Houston", "Phoenix",
                      "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"];
        let states = ["New York", "California", "Illinois", "Texas", "Arizona",
                      "Pennsylvania", "Texas", "California", "Texas", "Texas"];
        let companies = ["Acme", "Globex", "Initech", "Umbrella", "Cyberdyne",
                         "Soylent", "Aperture", "Tyrell", "Weyland", "Oscorp"];
        let departments = ["Engineering", "Sales", "Marketing", "Research", "Legal"];

        let mut triple_count = 0;

        // City → State mapping
        for (i, city) in cities.iter().enumerate() {
            n.remember(&format!("{city}:state"), states[i]);
            triple_count += 1;
        }

        // Company → HQ, Company → Industry
        for (i, company) in companies.iter().enumerate() {
            n.remember(&format!("{company}:hq"), cities[i % cities.len()]);
            n.remember(&format!("{company}:industry"), "Technology");
            n.remember(&format!("{company}:ceo"), &format!("CEO_{company}"));
            triple_count += 3;
        }

        // 100 employees, each with company, department, city
        for i in 0..100 {
            let name = format!("emp_{i:03}");
            let company = companies[i % companies.len()];
            let dept = departments[i % departments.len()];
            let city = cities[i % cities.len()];

            n.remember(&format!("{name}:worksAt"), company);
            n.remember(&format!("{name}:department"), dept);
            n.remember(&format!("{name}:livesIn"), city);
            n.remember(&format!("{name}:salary"), &format!("{}", 80000 + i * 1000));
            n.remember(&format!("{name}:manager"), &format!("mgr_{}", i / 10));
            triple_count += 5;
        }

        println!("  Stored {triple_count} triples\n");

        // Multi-hop queries
        println!("  ── Multi-hop queries ──\n");

        let mut correct = 0;
        let total = 20;

        for i in 0..total {
            let emp = format!("emp_{i:03}");
            let expected_company = companies[i % companies.len()];
            let expected_hq = cities[i % cities.len()];
            let expected_state = states[i % cities.len()];

            // 2-hop: employee → company → HQ
            let hop1 = n.recall(&format!("{emp}:worksAt"), "kg");
            let company = hop1.answer.as_deref().unwrap_or("?");

            let hop2 = n.recall(&format!("{company}:hq"), "kg");
            let hq = hop2.answer.as_deref().unwrap_or("?");

            // 3-hop: employee → company → HQ → state
            let hop3 = n.recall(&format!("{hq}:state"), "kg");
            let state = hop3.answer.as_deref().unwrap_or("?");

            let ok_2hop = company == expected_company && hq == expected_hq;
            let ok_3hop = ok_2hop && state == expected_state;

            if ok_3hop { correct += 1; }

            if i < 5 || !ok_3hop {
                let status = if ok_3hop { "✓".to_string() } else {
                    format!("✗ (expected {} → {} → {})", expected_company, expected_hq, expected_state)
                };
                println!("  {emp} → {company} → {hq} → {state}  {status}");
            }
        }

        if correct == total {
            println!("  ... (all correct)");
        }

        println!("\n  3-hop queries: {correct}/{total} correct");
        assert!(correct >= total - 2, "At least {}/{total} 3-hop queries should succeed", total - 2);
    }
}
