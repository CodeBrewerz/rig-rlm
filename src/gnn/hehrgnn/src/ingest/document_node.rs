//! Document Node Integration
//!
//! Connects vision pipeline output (document embeddings) to the
//! heterogeneous graph as "document" node type.
//!
//! Pipeline:
//!   Document image → DocumentVit → pooled embedding → DocumentAttachment
//!   DocumentAttachment → "document" node in HeteroGraph
//!   Connected via "attached_to" edges to tx/account/user nodes

use std::collections::HashMap;

use crate::model::vision::document_vit::{DocumentVit, DocumentVitConfig};

/// A document attachment linking a visual document to graph entities.
#[derive(Debug, Clone)]
pub struct DocumentAttachment {
    /// Unique document identifier.
    pub doc_id: String,
    /// Document type.
    pub doc_type: DocumentType,
    /// Pooled embedding from DocumentVit.
    pub embedding: Vec<f32>,
    /// Linked entity IDs: (node_type, entity_id).
    pub linked_entities: Vec<(String, String)>,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

/// Types of financial documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentType {
    /// Receipt from a purchase.
    Receipt,
    /// Bank statement (monthly/quarterly).
    BankStatement,
    /// Invoice or bill.
    Invoice,
    /// Tax document (W-2, 1099, etc.).
    TaxDocument,
    /// Other financial document.
    Other,
}

impl DocumentType {
    /// Default edge relation for this document type.
    pub fn edge_relation(&self) -> &str {
        match self {
            DocumentType::Receipt => "receipt_for",
            DocumentType::BankStatement => "statement_of",
            DocumentType::Invoice => "invoice_to",
            DocumentType::TaxDocument => "tax_doc_for",
            DocumentType::Other => "attached_to",
        }
    }
}

/// Document processor for batch document→embedding conversion.
#[derive(Debug)]
pub struct DocumentProcessor {
    /// The vision transformer model.
    pub vit: DocumentVit,
    /// Processed documents.
    pub documents: Vec<DocumentAttachment>,
}

impl DocumentProcessor {
    /// Create a new document processor with default config.
    pub fn new() -> Self {
        Self {
            vit: DocumentVit::new(DocumentVitConfig::default()),
            documents: Vec::new(),
        }
    }

    /// Create with custom ViT config.
    pub fn with_config(config: DocumentVitConfig) -> Self {
        Self {
            vit: DocumentVit::new(config),
            documents: Vec::new(),
        }
    }

    /// Process a document image and create an attachment.
    ///
    /// Returns the document embedding for graph integration.
    pub fn process_document(
        &mut self,
        doc_id: &str,
        doc_type: DocumentType,
        image: &[f32],
        height: usize,
        width: usize,
        linked_entities: Vec<(String, String)>,
    ) -> DocumentAttachment {
        let embedding = self.vit.encode_pooled(image, height, width);

        let attachment = DocumentAttachment {
            doc_id: doc_id.to_string(),
            doc_type,
            embedding,
            linked_entities,
            metadata: HashMap::new(),
        };

        self.documents.push(attachment.clone());
        attachment
    }

    /// Generate graph facts from processed documents.
    ///
    /// Returns: Vec of (src_type, src_id, relation, dst_type, dst_id)
    /// and a map of document embeddings for node features.
    pub fn to_graph_facts(&self) -> (Vec<GraphDocFact>, HashMap<String, Vec<f32>>) {
        let mut facts = Vec::new();
        let mut embeddings = HashMap::new();

        for doc in &self.documents {
            embeddings.insert(doc.doc_id.clone(), doc.embedding.clone());

            for (entity_type, entity_id) in &doc.linked_entities {
                facts.push(GraphDocFact {
                    src_type: "document".to_string(),
                    src_id: doc.doc_id.clone(),
                    relation: doc.doc_type.edge_relation().to_string(),
                    dst_type: entity_type.clone(),
                    dst_id: entity_id.clone(),
                });
            }
        }

        (facts, embeddings)
    }

    /// Set the ViT to inference mode (drop PE for generalization).
    pub fn set_inference(&mut self) {
        self.vit.set_inference();
    }

    /// Number of processed documents.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }
}

/// A fact linking a document node to another entity in the graph.
#[derive(Debug, Clone)]
pub struct GraphDocFact {
    pub src_type: String,
    pub src_id: String,
    pub relation: String,
    pub dst_type: String,
    pub dst_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_type_relations() {
        assert_eq!(DocumentType::Receipt.edge_relation(), "receipt_for");
        assert_eq!(DocumentType::BankStatement.edge_relation(), "statement_of");
        assert_eq!(DocumentType::Invoice.edge_relation(), "invoice_to");
        assert_eq!(DocumentType::TaxDocument.edge_relation(), "tax_doc_for");
    }

    #[test]
    fn test_process_document() {
        let config = DocumentVitConfig::tiny_test();
        let mut processor = DocumentProcessor::with_config(config);

        // Fake 8×8 grayscale receipt image
        let image = vec![0.5f32; 1 * 8 * 8];
        let attachment = processor.process_document(
            "receipt_001",
            DocumentType::Receipt,
            &image,
            8,
            8,
            vec![
                ("tx".to_string(), "tx_42".to_string()),
                ("merchant".to_string(), "starbucks".to_string()),
            ],
        );

        assert_eq!(attachment.doc_id, "receipt_001");
        assert_eq!(attachment.embedding.len(), 8); // tiny_test embed_dim
        assert_eq!(attachment.linked_entities.len(), 2);
    }

    #[test]
    fn test_to_graph_facts() {
        let config = DocumentVitConfig::tiny_test();
        let mut processor = DocumentProcessor::with_config(config);

        let image = vec![0.5f32; 1 * 8 * 8];
        processor.process_document(
            "receipt_001",
            DocumentType::Receipt,
            &image,
            8,
            8,
            vec![("tx".to_string(), "tx_42".to_string())],
        );
        processor.process_document(
            "statement_001",
            DocumentType::BankStatement,
            &image,
            8,
            8,
            vec![("account".to_string(), "acct_1".to_string())],
        );

        let (facts, embeddings) = processor.to_graph_facts();
        assert_eq!(facts.len(), 2);
        assert_eq!(embeddings.len(), 2);

        // Check fact relations match document types
        assert_eq!(facts[0].relation, "receipt_for");
        assert_eq!(facts[0].src_type, "document");
        assert_eq!(facts[0].dst_id, "tx_42");

        assert_eq!(facts[1].relation, "statement_of");
        assert_eq!(facts[1].dst_id, "acct_1");
    }

    #[test]
    fn test_document_processor_inference_mode() {
        let config = DocumentVitConfig::tiny_test();
        let mut processor = DocumentProcessor::with_config(config);
        processor.set_inference();

        // Should work on different sized documents in inference mode
        let img1 = vec![0.5f32; 1 * 8 * 8];
        let img2 = vec![0.5f32; 1 * 16 * 8];

        let a1 = processor.process_document("doc1", DocumentType::Receipt, &img1, 8, 8, vec![]);
        let a2 = processor.process_document("doc2", DocumentType::Invoice, &img2, 16, 8, vec![]);

        // Both produce same dimension embeddings (DroPE!)
        assert_eq!(a1.embedding.len(), a2.embedding.len());
    }

    #[test]
    fn test_batch_processing() {
        let config = DocumentVitConfig::tiny_test();
        let mut processor = DocumentProcessor::with_config(config);

        for i in 0..5 {
            let image = vec![i as f32 * 0.1; 1 * 8 * 8];
            processor.process_document(
                &format!("doc_{}", i),
                DocumentType::Receipt,
                &image,
                8,
                8,
                vec![("tx".to_string(), format!("tx_{}", i))],
            );
        }

        assert_eq!(processor.num_documents(), 5);
        let (facts, embeddings) = processor.to_graph_facts();
        assert_eq!(facts.len(), 5);
        assert_eq!(embeddings.len(), 5);
    }
}
