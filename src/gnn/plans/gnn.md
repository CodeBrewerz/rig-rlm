Here is a comprehensive, phased project plan designed specifically for a Rust development team. It breaks down the implementation of HEHRGNN into manageable sprints, from data ingestion to metric evaluation, leveraging the Burn framework.

### Phase 1: Project Setup & Dependency Management

**Goal:** Establish the Rust environment and Burn framework configurations.

* **Initialize Cargo Workspace:** Set up a standard Rust binary project.
* **Burn Configuration:** Add `burn` to `Cargo.toml`. The team should configure it to use the `Wgpu` backend for local GPU acceleration (or `Tch` if they prefer PyTorch bindings, though `Wgpu` is Burn's native strength) and `Autodiff` for training.
* **Data Wrangling Crates:** Include `polars` or `serde_json` for parsing the raw dataset files.

### Phase 2: Data Pipeline for Hyper-Relational KGs

**Goal:** Parse n-ary datasets and build Burn-compatible dataloaders. Standard Knowledge Graphs use triples (Head, Relation, Tail). Hyper-relational graphs add "qualifiers" (Key-Value pairs modifying the primary fact).

* **Dataset Acquisition:** Download standard n-ary KG datasets used in the paper: **WD50K**, **JF17K**, or **WikiPeople**.
* **Vocabulary Mapping:** Create string-to-integer dictionaries mapping every unique Entity and Relation to an ID ($0$ to $N-1$).
* **Batching Strategy:** The dev team must design a custom `burn::data::dataloader::DataLoader`. A single training batch should yield:
* `primary_triples`: Tensor of shape `[batch_size, 3]` (Head, Relation, Tail).
* `qualifiers`: Tensor containing the qualifier relations and entities attached to the primary triple. Since the number of qualifiers varies per fact, the team should use padding or a flattened sparse representation.


* **Negative Sampling:** Implement a function to generate corrupted facts. For every true hyperedge, generate negative samples by replacing the Head, Tail, or one of the Qualifier Entities with a random entity.

### Phase 3: Core HEHRGNN Architecture (Burn Implementation)

**Goal:** Implement the GNN module with the specific 3-step message passing.

* **Embedding Layers:** Initialize `Embedding` modules for Entities and Relations with a configurable `hidden_dim`.
* **Step 1 (Gather):** Implement the logic to pool (e.g., sum or mean) the embeddings of the primary Head, Tail, and all Qualifier Entities to create a `hyperedge_instance` representation.
* **Step 2 (Apply):** Concatenate or add the `hyperedge_instance` to the primary Relation embedding. Pass this through a `Linear` layer + ReLU activation to create the updated relation representation.
* **Step 3 (Scatter):** Project the updated representations back to the entity space. The team will need to carefully use Burn's `.scatter()` method to update the global node embeddings for the entities involved in the batch.

### Phase 4: Training Loop & Loss Function

**Goal:** Set up the optimizer, forward pass, and backpropagation.

* **Scoring Function:** Define how a hyperedge is scored. Usually, this is a translation-based function (like TransE adapted for hyperedges) or a semantic matching function (like DistMult), where the score is a scalar representing the plausibility of the fact.
* **Margin Ranking Loss:** Implement the loss function to push positive fact scores higher than negative fact scores by a specific margin $\gamma$.

$$\mathcal{L} = \sum_{f \in \mathcal{F}^+} \sum_{f' \in \mathcal{F}^-} \max(0, \gamma - score(f) + score(f'))$$


* **Optimizer:** Configure Burn's `AdamW` optimizer. Set up a learning rate scheduler (e.g., step decay).

### Phase 5: Evaluation & Metrics

**Goal:** Implement the evaluation protocol to test the model on the validation and test sets.

* **Link Prediction Protocol:** To test a fact, the model must hide one entity (e.g., the Head), replace it with *every* other entity in the vocabulary, score all of them, and rank them.
* **Filtered Setting:** Ensure the dev team implements "filtered" ranking. If a predicted fact happens to be another true fact in the training set, it shouldn't be penalized as a negative.
* **Metrics Calculation:** Implement Mean Reciprocal Rank (MRR) and Hits@K (where $K \in \{1, 3, 10\}$).

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$



### Phase 6: Result Reproduction & Hyperparameter Tuning

**Goal:** Match the performance reported in the HEHRGNN paper.

* **Paper Configurations:** Extract the exact hyperparameter grid from the paper (Embedding dimension, margin, learning rate, batch size, dropout rates).
* **Experiment Tracking:** Output training loss, validation MRR, and test metrics to a log file or a tracker.
* **Benchmarking:** Compare the Rust/Burn implementation's final MRR and Hits@10 on WD50K/JF17K against the exact numbers in the paper's results table.

---

Would you like me to draft a specific `struct` and `DataLoader` skeleton in Rust for Phase 2, so your team has a head start on parsing the n-ary graph data?