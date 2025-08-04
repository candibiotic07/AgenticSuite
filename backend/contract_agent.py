"""
Contract Review & Risk-Flagging Agent

A sophisticated, self-improving contract analysis agent that combines rule-based 
pattern matching with advanced LLM reasoning to identify and flag contractual risks.
"""

import os
import json
import logging
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import re
from collections import defaultdict, Counter

# Document Processing
import PyPDF2
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image

# Text Processing & Similarity
from rapidfuzz import fuzz, process
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Vector Database & Embeddings
import chromadb
from chromadb.config import Settings

# Google Gemini API
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Data Processing
import pandas as pd
import numpy as np

# Report Generation
from jinja2 import Template, Environment, FileSystemLoader

# WeasyPrint - Optional dependency for PDF generation
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    print(f"⚠️ WeasyPrint not available ({type(e).__name__}: {str(e)[:100]}...) - will use ReportLab fallback for PDF generation")

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Utilities
from tqdm import tqdm
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create main data directory
BASE_DATA_DIR = "contractDATAtemp"
os.makedirs(os.path.join(BASE_DATA_DIR, "logs"), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DATA_DIR, "logs", "contract_agent.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some features may be limited.")

@dataclass
class ClauseMetadata:
    """Metadata for contract clauses"""
    clause_id: str
    position: int
    section: str
    clause_type: str
    confidence: float
    fingerprint: str

@dataclass
class RiskAssessment:
    """Risk assessment result for a clause"""
    risk_level: str  # HIGH, MEDIUM, LOW
    confidence: float
    rationale: str
    rule_matches: List[str]
    suggestions: List[str]
    priority_score: float

@dataclass
class ContractClause:
    """Individual contract clause with analysis"""
    text: str
    metadata: ClauseMetadata
    risk_assessment: Optional[RiskAssessment] = None
    user_feedback: Optional[Dict] = None

@dataclass
class ContractAnalysis:
    """Complete contract analysis results"""
    document_id: str
    clauses: List[ContractClause]
    executive_summary: Dict[str, Any]
    generated_at: datetime
    processing_stats: Dict[str, Any]

class DocumentProcessor:
    """Handle document extraction from various formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with fallback to OCR"""
        try:
            # First try with pdfplumber (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return text
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                if text.strip():
                    return text
            
            # Final fallback to OCR (if text extraction fails)
            logger.warning("Text extraction failed, attempting OCR...")
            return self._ocr_fallback(file_path)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return self._ocr_fallback(file_path)
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def _ocr_fallback(self, file_path: str) -> str:
        """OCR fallback for scanned documents"""
        try:
            # This is a basic OCR implementation
            # In production, you might want to use more sophisticated OCR
            text = pytesseract.image_to_string(file_path)
            return text
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Main text extraction method"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif suffix == '.docx':
            return self.extract_text_from_docx(str(file_path))
        elif suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

class ClauseExtractor:
    """Enhanced clause segmentation and type detection"""
    
    def __init__(self):
        self.clause_types = {
            'payment': ['payment', 'fee', 'cost', 'price', 'invoice', 'billing', 'compensation', 'salary', 'wage'],
            'liability': ['liable', 'liability', 'indemnify', 'indemnification', 'damages', 'loss', 'harm'],
            'termination': ['terminate', 'termination', 'breach', 'default', 'expire', 'dissolution', 'cancel'],
            'intellectual_property': ['intellectual property', 'copyright', 'patent', 'trademark', 'trade secret', 'ip'],
            'confidentiality': ['confidential', 'non-disclosure', 'proprietary', 'secret', 'confidentiality', 'nda'],
            'force_majeure': ['force majeure', 'act of god', 'unforeseeable', 'beyond control', 'extraordinary'],
            'governing_law': ['governing law', 'jurisdiction', 'dispute resolution', 'arbitration', 'court', 'legal'],
            'warranties': ['warrant', 'warranty', 'guarantee', 'representation', 'assurance', 'promise'],
            'indemnification': ['indemnify', 'indemnification', 'hold harmless', 'defend', 'protect'],
            'assignment': ['assign', 'assignment', 'transfer', 'delegate', 'succession'],
            'amendment': ['amend', 'amendment', 'modify', 'modification', 'change', 'alter'],
            'entire_agreement': ['entire agreement', 'complete agreement', 'supersede', 'merger clause'],
            'severability': ['severability', 'severable', 'invalid', 'unenforceable', 'remainder'],
            'notices': ['notice', 'notification', 'communicate', 'inform', 'written notice'],
            'dispute_resolution': ['dispute', 'resolution', 'mediation', 'arbitration', 'litigation'],
            'limitation_of_liability': ['limitation of liability', 'limited liability', 'cap on damages', 'exclude'],
            'general': ['general', 'miscellaneous', 'other', 'various']
        }
        
        # Clause boundary indicators
        self.section_patterns = [
            r'\b\d+\.\s*',  # 1. Section
            r'\b\([a-z]\)\s*',  # (a) Subsection
            r'\b[A-Z][A-Z\s]+:',  # TITLE:
            r'\bSection\s+\d+',  # Section 1
            r'\bArticle\s+\d+',  # Article 1
        ]
    
    def segment_into_clauses(self, text: str) -> List[str]:
        """Segment document into logical clauses"""
        # Clean and normalize text
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Split by strong section indicators
        clauses = []
        current_clause = ""
        
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Check if this sentence starts a new section
            is_new_section = any(re.match(pattern, sentence.strip()) for pattern in self.section_patterns)
            
            if is_new_section and current_clause.strip():
                clauses.append(current_clause.strip())
                current_clause = sentence
            else:
                current_clause += " " + sentence
        
        # Add the last clause
        if current_clause.strip():
            clauses.append(current_clause.strip())
        
        # Filter out very short clauses (likely headers)
        clauses = [clause for clause in clauses if len(clause.split()) > 10]
        
        return clauses
    
    def detect_clause_type(self, clause_text: str) -> Tuple[str, float]:
        """Detect the type of a clause with confidence"""
        clause_lower = clause_text.lower()
        type_scores = {}
        
        for clause_type, keywords in self.clause_types.items():
            score = 0
            for keyword in keywords:
                if keyword in clause_lower:
                    # Give higher score for exact matches
                    score += 1
                    # Bonus for multiple occurrences
                    score += clause_lower.count(keyword) * 0.5
            
            if score > 0:
                type_scores[clause_type] = score
        
        if not type_scores:
            return 'general', 0.3
        
        best_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[best_type]
        
        # Normalize confidence (rough heuristic)
        confidence = min(max_score / 3.0, 1.0)
        
        return best_type, confidence
    
    def create_clause_fingerprint(self, clause_text: str) -> str:
        """Create a unique fingerprint for a clause"""
        # Remove common words and normalize
        words = word_tokenize(clause_text.lower())
        try:
            stop_words = set(stopwords.words('english'))
            significant_words = [w for w in words if w.isalpha() and w not in stop_words]
        except:
            significant_words = [w for w in words if w.isalpha()]
        
        # Create hash of significant words
        text_hash = hashlib.md5(' '.join(sorted(significant_words[:20])).encode()).hexdigest()
        return text_hash[:12]
    
    def llm_extract_clauses(self, text: str) -> List[Dict]:
        """Use LLM to intelligently extract and classify clauses"""
        try:
            prompt = f"""
You are a legal expert specializing in contract analysis. Please analyze the following contract text and extract individual clauses. 

For each clause you identify, provide:
1. The exact text of the clause
2. The clause type from this list: payment, liability, termination, intellectual_property, confidentiality, force_majeure, governing_law, warranties, indemnification, assignment, amendment, entire_agreement, severability, notices, dispute_resolution, limitation_of_liability, general

Instructions:
- A clause is a distinct legal provision with a specific purpose
- Include all important contractual provisions as separate clauses
- Don't include boilerplate text like headers, signatures, or purely administrative content
- If a long paragraph contains multiple distinct legal concepts, split it into separate clauses
- Each clause should be meaningful and contain actionable legal language

Return your response as a JSON array where each item has:
- "text": the exact clause text
- "type": the clause type from the list above
- "confidence": your confidence in the classification (0.0-1.0)

Contract text:
{text}

JSON Response:
"""

            # Use Gemini for clause extraction
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_output_tokens=64000
                )
            )
            
            if response.text:
                try:
                    # Try to parse JSON response
                    import json
                    clauses_data = json.loads(response.text)
                    return clauses_data
                except json.JSONDecodeError:
                    # Fallback: try to extract JSON from response
                    import re
                    json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                    if json_match:
                        clauses_data = json.loads(json_match.group())
                        return clauses_data
                    else:
                        logger.warning(f"Could not parse LLM response as JSON: {response.text[:200]}...")
                        return None
            else:
                logger.warning("No response from LLM for clause extraction")
                return None
                
        except Exception as e:
            logger.error(f"LLM clause extraction failed: {e}")
            return None

    def extract_clauses(self, text: str, document_id: str) -> List[ContractClause]:
        """Main method to extract and analyze clauses"""
        # First try LLM-based extraction
        llm_clauses = self.llm_extract_clauses(text)
        
        if llm_clauses and len(llm_clauses) > 1:
            logger.info(f"LLM extracted {len(llm_clauses)} clauses")
            clauses = []
            
            for i, clause_data in enumerate(llm_clauses):
                clause_text = clause_data.get('text', '')
                clause_type = clause_data.get('type', 'general')
                confidence = float(clause_data.get('confidence', 0.5))
                
                # Clean up clause text
                clause_text = clause_text.strip()
                if len(clause_text) < 20:  # Skip very short clauses
                    continue
                    
                fingerprint = self.create_clause_fingerprint(clause_text)
                
                metadata = ClauseMetadata(
                    clause_id=f"{document_id}_clause_{i:03d}",
                    position=i,
                    section=f"Clause {i+1}",
                    clause_type=clause_type,
                    confidence=confidence,
                    fingerprint=fingerprint
                )
                
                clause = ContractClause(
                    text=clause_text,
                    metadata=metadata
                )
                
                clauses.append(clause)
        else:
            # Fallback to regex-based extraction
            logger.info("Falling back to regex-based clause extraction")
            clause_texts = self.segment_into_clauses(text)
            clauses = []
            
            for i, clause_text in enumerate(clause_texts):
                clause_type, confidence = self.detect_clause_type(clause_text)
                fingerprint = self.create_clause_fingerprint(clause_text)
            
                metadata = ClauseMetadata(
                    clause_id=f"{document_id}_clause_{i:03d}",
                    position=i,
                    section=f"Section {i+1}",
                    clause_type=clause_type,
                    confidence=confidence,
                    fingerprint=fingerprint
                )
                
                clause = ContractClause(
                    text=clause_text,
                    metadata=metadata
                )
                
                clauses.append(clause)
        
        return clauses

class VectorStoreManager:
    """Manage ChromaDB vector store for semantic retrieval"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = os.path.join(BASE_DATA_DIR, "vectorstore")
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize collections
        self.clauses_collection = self._get_or_create_collection("contract_clauses")
        self.feedback_collection = self._get_or_create_collection("user_feedback")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API"""
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fallback to simple vector (in production, use better fallback)
            return [0.0] * 768
    
    def add_clause(self, clause: ContractClause, embedding: Optional[List[float]] = None):
        """Add a clause to the vector store"""
        if embedding is None:
            embedding = self.generate_embedding(clause.text)
        
        metadata = {
            'clause_id': clause.metadata.clause_id,
            'clause_type': clause.metadata.clause_type,
            'fingerprint': clause.metadata.fingerprint,
            'added_at': datetime.now().isoformat()
        }
        
        self.clauses_collection.add(
            embeddings=[embedding],
            documents=[clause.text],
            metadatas=[metadata],
            ids=[clause.metadata.clause_id]
        )
    
    def search_similar_clauses(self, query_text: str, clause_type: str = None, n_results: int = 5) -> List[Dict]:
        """Search for semantically similar clauses"""
        query_embedding = self.generate_embedding(query_text)
        
        where_filter = {}
        if clause_type:
            where_filter['clause_type'] = clause_type
        
        results = self.clauses_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return [{
            'text': doc,
            'metadata': meta,
            'distance': dist
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )]
    
    def add_feedback(self, clause_fingerprint: str, feedback_data: Dict):
        """Add user feedback to vector store"""
        feedback_id = f"feedback_{clause_fingerprint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feedback_data.update({
            'clause_fingerprint': clause_fingerprint,
            'timestamp': datetime.now().isoformat()
        })
        
        # Create embedding for feedback context
        feedback_text = f"{feedback_data.get('original_text', '')} {feedback_data.get('correction_reason', '')}"
        embedding = self.generate_embedding(feedback_text)
        
        self.feedback_collection.add(
            embeddings=[embedding],
            documents=[json.dumps(feedback_data)],
            metadatas=[{'clause_fingerprint': clause_fingerprint, 'type': 'feedback'}],
            ids=[feedback_id]
        )

class RiskAnalyzer:
    """Hybrid rule-based and LLM risk analysis"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.risk_registry = self._load_risk_registry()
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Safety settings for contract analysis
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    def _load_risk_registry(self) -> Dict:
        """Load JSON risk registry (create default if not exists)"""
        data_dir = os.path.join(BASE_DATA_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        registry_path = os.path.join(data_dir, 'risk_registry.json')
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                return json.load(f)
        
        # Create default risk registry
        default_registry = {
            "payment": {
                "high_risk": [
                    "payment due immediately",
                    "no refund",
                    "penalty for late payment",
                    "interest rate above 15%",
                    "payment in advance"
                ],
                "medium_risk": [
                    "payment within 30 days",
                    "late payment fee",
                    "escrow required"
                ]
            },
            "liability": {
                "high_risk": [
                    "unlimited liability",
                    "personal liability",
                    "no limitation of liability",
                    "consequential damages",
                    "punitive damages"
                ],
                "medium_risk": [
                    "liability cap above $100,000",
                    "indemnification required"
                ]
            },
            "termination": {
                "high_risk": [
                    "termination without cause",
                    "immediate termination",
                    "no cure period",
                    "termination for convenience"
                ],
                "medium_risk": [
                    "30 days notice required",
                    "material breach"
                ]
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(default_registry, f, indent=2)
        
        return default_registry
    
    def apply_rule_based_analysis(self, clause: ContractClause) -> Tuple[List[str], str]:
        """Apply rule-based risk detection"""
        clause_type = clause.metadata.clause_type
        clause_text_lower = clause.text.lower()
        
        matches = []
        max_risk_level = "LOW"
        
        if clause_type in self.risk_registry:
            rules = self.risk_registry[clause_type]
            
            # Check high-risk patterns
            if "high_risk" in rules:
                for pattern in rules["high_risk"]:
                    similarity = fuzz.partial_ratio(pattern.lower(), clause_text_lower)
                    if similarity > 80:  # High similarity threshold
                        matches.append(f"HIGH: {pattern} (similarity: {similarity}%)")
                        max_risk_level = "HIGH"
            
            # Check medium-risk patterns
            if "medium_risk" in rules and max_risk_level != "HIGH":
                for pattern in rules["medium_risk"]:
                    similarity = fuzz.partial_ratio(pattern.lower(), clause_text_lower)
                    if similarity > 75:  # Medium similarity threshold
                        matches.append(f"MEDIUM: {pattern} (similarity: {similarity}%)")
                        if max_risk_level != "HIGH":
                            max_risk_level = "MEDIUM"
        
        return matches, max_risk_level
    
    def generate_llm_analysis(self, clause: ContractClause, rule_matches: List[str], similar_clauses: List[Dict]) -> Dict:
        """Generate LLM-based risk analysis with self-consistency"""
        # Prepare context
        context = self._prepare_analysis_context(clause, rule_matches, similar_clauses)
        
        # Self-consistency loop (3-5 iterations)
        analyses = []
        for i in range(3):
            try:
                analysis = self._single_llm_analysis(context, iteration=i)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"LLM analysis iteration {i} failed: {e}")
        
        # Majority voting and confidence aggregation
        return self._aggregate_analyses(analyses)
    
    def _prepare_analysis_context(self, clause: ContractClause, rule_matches: List[str], similar_clauses: List[Dict]) -> str:
        """Prepare context for LLM analysis"""
        context = f"""
You are a expert contract lawyer analyzing contractual risks. 

CLAUSE TO ANALYZE:
Type: {clause.metadata.clause_type}
Text: {clause.text}

RULE-BASED MATCHES:
{chr(10).join(rule_matches) if rule_matches else "No exact rule matches found."}

SIMILAR PAST CLAUSES:
"""
        
        for i, similar in enumerate(similar_clauses[:3], 1):
            context += f"\n{i}. (Distance: {similar['distance']:.3f}) {similar['text'][:200]}..."
        
        context += """

ANALYSIS INSTRUCTIONS:
1. Assess the risk level: HIGH, MEDIUM, or LOW
2. Consider both explicit terms and implicit risks
3. Focus on potential negative impacts to the contracting party
4. Provide specific rationale for your assessment
5. Suggest alternative language if risks are identified
6. Rate your confidence (0.0-1.0) in this assessment

Respond in JSON format:
{
    "risk_level": "HIGH|MEDIUM|LOW",
    "confidence": 0.85,
    "rationale": "Detailed explanation...",
    "specific_risks": ["risk1", "risk2"],
    "suggestions": ["suggestion1", "suggestion2"]
}
"""
        return context
    
    def _single_llm_analysis(self, context: str, iteration: int) -> Dict:
        """Single LLM analysis iteration"""
        # Add slight variation for self-consistency
        variation_prompts = [
            "Carefully analyze this contract clause for risks:",
            "As a legal expert, evaluate the following clause:",
            "Review this contractual provision for potential issues:"
        ]
        
        prompt = f"{variation_prompts[iteration % len(variation_prompts)]}\n\n{context}"
        
        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            
            # Parse JSON response
            try:
                # Extract JSON from response
                text = response.text
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = text[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                # Validate and clean the parsed data
                validated_data = {
                    "risk_level": str(parsed_data.get('risk_level', 'MEDIUM')).upper(),
                    "confidence": float(parsed_data.get('confidence', 0.5)),
                    "rationale": str(parsed_data.get('rationale', 'No rationale provided')),
                    "specific_risks": [],
                    "suggestions": []
                }
                
                # Ensure risk_level is valid
                if validated_data["risk_level"] not in ['HIGH', 'MEDIUM', 'LOW']:
                    validated_data["risk_level"] = 'MEDIUM'
                
                # Ensure confidence is between 0 and 1
                validated_data["confidence"] = max(0.0, min(1.0, validated_data["confidence"]))
                
                # Handle specific_risks - ensure they are strings
                specific_risks = parsed_data.get('specific_risks', [])
                if isinstance(specific_risks, list):
                    for risk in specific_risks:
                        if isinstance(risk, str):
                            validated_data["specific_risks"].append(risk)
                        else:
                            validated_data["specific_risks"].append(str(risk))
                
                # Handle suggestions - ensure they are strings
                suggestions = parsed_data.get('suggestions', [])
                if isinstance(suggestions, list):
                    for suggestion in suggestions:
                        if isinstance(suggestion, str):
                            validated_data["suggestions"].append(suggestion)
                        else:
                            validated_data["suggestions"].append(str(suggestion))
                
                return validated_data
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"JSON parsing failed in LLM analysis: {e}")
                # Fallback parsing
                return {
                    "risk_level": "MEDIUM",
                    "confidence": 0.5,
                    "rationale": response.text[:500] if hasattr(response, 'text') else "Analysis failed",
                    "specific_risks": [],
                    "suggestions": []
                }
                
        except Exception as e:
            logger.error(f"LLM analysis iteration failed: {e}")
            return {
                "risk_level": "MEDIUM",
                "confidence": 0.3,
                "rationale": f"Analysis failed: {str(e)}",
                "specific_risks": [],
                "suggestions": []
            }
    
    def _aggregate_analyses(self, analyses: List[Dict]) -> Dict:
        """Aggregate multiple analysis results using majority voting"""
        if not analyses:
            return {
                "risk_level": "MEDIUM",
                "confidence": 0.3,
                "rationale": "Analysis failed",
                "specific_risks": [],
                "suggestions": []
            }
        
        # Majority vote on risk level
        risk_levels = [a.get('risk_level', 'MEDIUM') for a in analyses]
        risk_level = Counter(risk_levels).most_common(1)[0][0]
        
        # Average confidence
        confidences = [a.get('confidence', 0.5) for a in analyses]
        avg_confidence = sum(confidences) / len(confidences)
        
        # Combine rationales and suggestions
        rationales = [a.get('rationale', '') for a in analyses]
        all_suggestions = []
        all_specific_risks = []
        
        for a in analyses:
            # Handle suggestions - ensure they are strings
            suggestions = a.get('suggestions', [])
            if isinstance(suggestions, list):
                for s in suggestions:
                    if isinstance(s, str):
                        all_suggestions.append(s)
                    elif isinstance(s, dict):
                        # Convert dict to string representation
                        all_suggestions.append(str(s))
                    else:
                        all_suggestions.append(str(s))
            
            # Handle specific_risks - ensure they are strings
            specific_risks = a.get('specific_risks', [])
            if isinstance(specific_risks, list):
                for r in specific_risks:
                    if isinstance(r, str):
                        all_specific_risks.append(r)
                    elif isinstance(r, dict):
                        # Convert dict to string representation
                        all_specific_risks.append(str(r))
                    else:
                        all_specific_risks.append(str(r))
        
        # Remove duplicates by converting to set after ensuring all elements are strings
        unique_suggestions = list(set(all_suggestions)) if all_suggestions else []
        unique_risks = list(set(all_specific_risks)) if all_specific_risks else []
        
        return {
            "risk_level": risk_level,
            "confidence": avg_confidence,
            "rationale": " | ".join(rationales),
            "specific_risks": unique_risks,
            "suggestions": unique_suggestions
        }
    
    def analyze_clause(self, clause: ContractClause) -> RiskAssessment:
        """Main clause analysis method"""
        # 1. Rule-based analysis
        rule_matches, rule_risk_level = self.apply_rule_based_analysis(clause)
        
        # 2. Semantic retrieval
        similar_clauses = self.vector_store.search_similar_clauses(
            clause.text, 
            clause.metadata.clause_type,
            n_results=5
        )
        
        # 3. LLM analysis with self-consistency
        llm_analysis = self.generate_llm_analysis(clause, rule_matches, similar_clauses)
        
        # 4. Combine results
        final_risk_level = self._combine_risk_levels(rule_risk_level, llm_analysis['risk_level'])
        
        # 5. Calculate priority score
        confidence = llm_analysis['confidence']
        risk_weight = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        priority_score = risk_weight[final_risk_level] * (1 - confidence)
        
        return RiskAssessment(
            risk_level=final_risk_level,
            confidence=confidence,
            rationale=llm_analysis['rationale'],
            rule_matches=rule_matches,
            suggestions=llm_analysis['suggestions'],
            priority_score=priority_score
        )
    
    def _combine_risk_levels(self, rule_level: str, llm_level: str) -> str:
        """Combine rule-based and LLM risk levels"""
        risk_hierarchy = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        # Take the higher risk level
        rule_weight = risk_hierarchy.get(rule_level, 1)
        llm_weight = risk_hierarchy.get(llm_level, 1)
        
        if rule_weight >= llm_weight:
            return rule_level
        else:
            return llm_level

class FeedbackManager:
    """Handle user feedback and preference learning"""
    
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.feedback_history = self._load_feedback_history()
    
    def _load_feedback_history(self) -> List[Dict]:
        """Load feedback history from file"""
        data_dir = os.path.join(BASE_DATA_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        feedback_file = os.path.join(data_dir, 'feedback_history.json')
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback_history(self):
        """Save feedback history to file"""
        data_dir = os.path.join(BASE_DATA_DIR, "data")
        feedback_file = os.path.join(data_dir, 'feedback_history.json')
        with open(feedback_file, 'w') as f:
            json.dump(self.feedback_history, f, indent=2)
    
    def collect_feedback(self, clause: ContractClause, user_correction: Dict) -> Dict:
        """Collect and process user feedback"""
        feedback_data = {
            'clause_fingerprint': clause.metadata.fingerprint,
            'original_assessment': asdict(clause.risk_assessment) if clause.risk_assessment else None,
            'user_correction': user_correction,
            'timestamp': datetime.now().isoformat(),
            'clause_type': clause.metadata.clause_type,
            'clause_text': clause.text[:500],  # Store snippet for context
        }
        
        # Add to history
        self.feedback_history.append(feedback_data)
        self._save_feedback_history()
        
        # Add to vector store
        self.vector_store.add_feedback(clause.metadata.fingerprint, feedback_data)
        
        return feedback_data
    
    def get_prioritized_clauses(self, clauses: List[ContractClause]) -> List[ContractClause]:
        """Sort clauses by priority score for feedback collection"""
        # Filter clauses with risk assessments and sort by priority
        assessed_clauses = [c for c in clauses if c.risk_assessment is not None]
        return sorted(assessed_clauses, key=lambda c: c.risk_assessment.priority_score, reverse=True)
    
    def infer_preferences(self) -> Dict[str, Any]:
        """PRELUDE-style preference inference from feedback history"""
        if len(self.feedback_history) < 5:
            return {}
        
        preferences = {
            'risk_tolerance': {},
            'clause_patterns': {},
            'correction_patterns': []
        }
        
        # Analyze risk tolerance by clause type
        for clause_type in ['payment', 'liability', 'termination', 'intellectual_property']:
            type_feedback = [f for f in self.feedback_history if f['clause_type'] == clause_type]
            
            if len(type_feedback) >= 3:
                risk_adjustments = []
                for feedback in type_feedback:
                    if feedback['original_assessment'] and feedback['user_correction']:
                        original_risk = feedback['original_assessment']['risk_level']
                        corrected_risk = feedback['user_correction'].get('risk_level')
                        
                        if corrected_risk and original_risk != corrected_risk:
                            risk_adjustments.append({
                                'from': original_risk,
                                'to': corrected_risk,
                                'reason': feedback['user_correction'].get('reason', '')
                            })
                
                if risk_adjustments:
                    preferences['risk_tolerance'][clause_type] = risk_adjustments
        
        # Identify common correction patterns
        correction_reasons = [f['user_correction'].get('reason', '') for f in self.feedback_history 
                            if f['user_correction'] and f['user_correction'].get('reason')]
        
        if correction_reasons:
            # Simple pattern detection (could be more sophisticated)
            reason_counts = Counter(correction_reasons)
            preferences['correction_patterns'] = reason_counts.most_common(5)
        
        return preferences

class ReportGenerator:
    """Generate professional HTML and PDF reports"""
    
    def __init__(self):
        self.template_dir = self._setup_templates()
    
    def _setup_templates(self) -> str:
        """Setup Jinja2 templates directory"""
        template_dir = os.path.join(BASE_DATA_DIR, "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # Create default HTML template if it doesn't exist
        template_path = os.path.join(template_dir, "contract_report.html")
        if not os.path.exists(template_path):
            self._create_default_template(template_path)
        
        return template_dir
    
    def _create_default_template(self, template_path: str):
        """Create default HTML template"""
        template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Contract Risk Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        .executive-summary { background-color: #e8f4f8; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .risk-high { background-color: #ffebee; border-left: 4px solid #f44336; }
        .risk-medium { background-color: #fff3e0; border-left: 4px solid #ff9800; }
        .risk-low { background-color: #e8f5e8; border-left: 4px solid #4caf50; }
        .clause { margin: 15px 0; padding: 15px; border-radius: 5px; }
        .confidence { font-size: 0.9em; color: #666; }
        .suggestions { background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 3px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Contract Risk Analysis Report</h1>
        <p><strong>Document:</strong> {{ analysis.document_id }}</p>
        <p><strong>Generated:</strong> {{ analysis.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p><strong>Total Clauses:</strong> {{ analysis.clauses|length }}</p>
    </div>
    
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <table>
            <tr>
                <th>Risk Level</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for level, count in analysis.executive_summary.risk_distribution.items() %}
            <tr>
                <td>{{ level }}</td>
                <td>{{ count }}</td>
                <td>{{ "%.1f"|format((count / analysis.clauses|length) * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
        
        <h3>Key Findings</h3>
        <ul>
            {% for finding in analysis.executive_summary.key_findings %}
            <li>{{ finding }}</li>
            {% endfor %}
        </ul>
    </div>
    
    <h2>Detailed Clause Analysis</h2>
    {% for clause in analysis.clauses %}
    {% if clause.risk_assessment %}
    <div class="clause risk-{{ clause.risk_assessment.risk_level.lower() }}">
        <h3>{{ clause.metadata.section }} - {{ clause.metadata.clause_type|title }}</h3>
        <div class="confidence">
            Risk: {{ clause.risk_assessment.risk_level }} 
            (Confidence: {{ "%.1f"|format(clause.risk_assessment.confidence * 100) }}%)
            Priority Score: {{ "%.2f"|format(clause.risk_assessment.priority_score) }}
        </div>
        
        <p><strong>Text:</strong> {{ clause.text[:300] }}{% if clause.text|length > 300 %}...{% endif %}</p>
        
        <p><strong>Risk Assessment:</strong> {{ clause.risk_assessment.rationale }}</p>
        
        {% if clause.risk_assessment.rule_matches %}
        <p><strong>Rule Matches:</strong></p>
        <ul>
            {% for match in clause.risk_assessment.rule_matches %}
            <li>{{ match }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if clause.risk_assessment.suggestions %}
        <div class="suggestions">
            <strong>Suggested Improvements:</strong>
            <ul>
                {% for suggestion in clause.risk_assessment.suggestions %}
                <li>{{ suggestion }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
    {% endif %}
    {% endfor %}
    
    <div class="header">
        <h2>Processing Statistics</h2>
        <ul>
            {% for key, value in analysis.processing_stats.items() %}
            <li><strong>{{ key|replace('_', ' ')|title }}:</strong> {{ value }}</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
        """
        
        with open(template_path, 'w') as f:
            f.write(template_content)
    
    def generate_html_report(self, analysis: ContractAnalysis) -> str:
        """Generate HTML report"""
        env = Environment(loader=FileSystemLoader(self.template_dir))
        template = env.get_template("contract_report.html")
        
        html_content = template.render(analysis=analysis)
        
        # Save HTML report in reports directory
        reports_dir = os.path.join(BASE_DATA_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = os.path.join(reports_dir, f"contract_report_{analysis.document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_filename
    
    def generate_pdf_report(self, html_file: str) -> str:
        """Convert HTML report to PDF"""
        if not WEASYPRINT_AVAILABLE:
            logger.info("WeasyPrint not available, using ReportLab fallback for PDF generation")
            return self._fallback_pdf_generation(html_file)
        
        try:
            # Ensure the PDF is saved in the reports directory
            reports_dir = os.path.join(BASE_DATA_DIR, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Get just the filename and add to reports directory
            html_basename = os.path.basename(html_file)
            pdf_filename = os.path.join(reports_dir, html_basename.replace('.html', '.pdf'))
            
            logger.info(f"Attempting to generate PDF using WeasyPrint: {pdf_filename}")
            weasyprint.HTML(filename=html_file).write_pdf(pdf_filename)
            logger.info("✅ PDF generated successfully using WeasyPrint")
            return pdf_filename
        except Exception as e:
            logger.warning(f"⚠️ WeasyPrint PDF generation failed: {e}")
            logger.info("Falling back to enhanced ReportLab PDF generation...")
            return self._enhanced_fallback_pdf_generation(html_file)
    
    def _fallback_pdf_generation(self, html_file: str) -> str:
        """Simple fallback PDF generation using reportlab"""
        # Ensure the PDF is also saved in the reports directory
        reports_dir = os.path.join(BASE_DATA_DIR, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Get just the filename and add to reports directory
        html_basename = os.path.basename(html_file)
        pdf_filename = os.path.join(reports_dir, html_basename.replace('.html', '_simple_fallback.pdf'))
        
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.darkblue,
            spaceAfter=20
        )
        
        story.append(Paragraph("Contract Risk Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Add simple content
        story.append(Paragraph("This is a simplified PDF report generated as fallback.", styles['Normal']))
        story.append(Paragraph(f"Generated from: {html_file}", styles['Normal']))
        story.append(Paragraph("Note: For full report content, please view the HTML version.", styles['Normal']))
        
        doc.build(story)
        return pdf_filename
    
    def _enhanced_fallback_pdf_generation(self, html_file: str) -> str:
        """Enhanced fallback PDF generation that extracts data from HTML"""
        try:
            # Read the HTML file to extract data
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Ensure the PDF is saved in the reports directory
            reports_dir = os.path.join(BASE_DATA_DIR, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Get just the filename and add to reports directory
            html_basename = os.path.basename(html_file)
            pdf_filename = os.path.join(reports_dir, html_basename.replace('.html', '_enhanced_fallback.pdf'))
            
            doc = SimpleDocTemplate(pdf_filename, pagesize=letter, topMargin=50, bottomMargin=50)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.darkblue,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue,
                spaceAfter=15,
                spaceBefore=20
            )
            
            # Add title
            story.append(Paragraph("Contract Risk Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Extract document info from HTML
            import re
            doc_match = re.search(r'<strong>Document:</strong>\s*([^<]+)', html_content)
            date_match = re.search(r'<strong>Generated:</strong>\s*([^<]+)', html_content)
            clauses_match = re.search(r'<strong>Total Clauses:</strong>\s*([^<]+)', html_content)
            
            if doc_match:
                story.append(Paragraph(f"<b>Document:</b> {doc_match.group(1).strip()}", styles['Normal']))
            if date_match:
                story.append(Paragraph(f"<b>Generated:</b> {date_match.group(1).strip()}", styles['Normal']))
            if clauses_match:
                story.append(Paragraph(f"<b>Total Clauses:</b> {clauses_match.group(1).strip()}", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", heading_style))
            
            # Extract risk distribution
            risk_data = []
            risk_pattern = r'<td>([^<]+)</td>\s*<td>(\d+)</td>\s*<td>([^<]+)</td>'
            risk_matches = re.findall(risk_pattern, html_content)
            
            if risk_matches:
                # Create table data
                table_data = [['Risk Level', 'Count', 'Percentage']]
                for level, count, percentage in risk_matches:
                    table_data.append([level.strip(), count.strip(), percentage.strip()])
                
                # Create table
                table = Table(table_data, colWidths=[2*inch, 1*inch, 1.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                story.append(Spacer(1, 20))
            
            # Extract key findings
            findings_pattern = r'<li>([^<]+)</li>'
            findings_matches = re.findall(findings_pattern, html_content)
            
            if findings_matches:
                story.append(Paragraph("Key Findings", heading_style))
                for finding in findings_matches[:5]:  # Limit to first 5 findings
                    story.append(Paragraph(f"• {finding.strip()}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Clause Analysis
            story.append(Paragraph("Detailed Clause Analysis", heading_style))
            
            # Extract clause information
            clause_pattern = r'<h3>([^<]+)</h3>.*?Risk:\s*([^<(]+).*?<p><strong>Text:</strong>\s*([^<]+)'
            clause_matches = re.findall(clause_pattern, html_content, re.DOTALL)
            
            for i, (section, risk, text) in enumerate(clause_matches[:10]):  # Limit to first 10 clauses
                story.append(Paragraph(f"<b>{section.strip()}</b>", styles['Heading3']))
                story.append(Paragraph(f"<b>Risk Level:</b> {risk.strip()}", styles['Normal']))
                story.append(Paragraph(f"<b>Text:</b> {text.strip()[:300]}{'...' if len(text.strip()) > 300 else ''}", styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Add note about full report
            story.append(Spacer(1, 30))
            story.append(Paragraph("Note: This is an enhanced PDF generated using ReportLab fallback. For the complete interactive report with all features, please view the HTML version.", styles['Italic']))
            
            doc.build(story)
            logger.info(f"✅ Enhanced fallback PDF generated: {pdf_filename}")
            return pdf_filename
            
        except Exception as e:
            logger.error(f"Enhanced fallback PDF generation failed: {e}")
            return self._fallback_pdf_generation(html_file)

class ContractAgent:
    """Main contract analysis agent orchestrator"""
    
    def __init__(self, persist_directory: str = None):
        """Initialize the contract agent"""
        if persist_directory is None:
            persist_directory = os.path.join(BASE_DATA_DIR, "agent_data")
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.clause_extractor = ClauseExtractor()
        self.vector_store = VectorStoreManager(
            os.path.join(BASE_DATA_DIR, "vectorstore")
        )
        self.risk_analyzer = RiskAnalyzer(self.vector_store)
        self.feedback_manager = FeedbackManager(self.vector_store)
        self.report_generator = ReportGenerator()
        
        logger.info("Contract Agent initialized successfully")
    
    def analyze_contract(self, file_path: str, document_id: Optional[str] = None) -> ContractAnalysis:
        """Main contract analysis workflow"""
        start_time = datetime.now()
        
        if document_id is None:
            document_id = Path(file_path).stem
        
        logger.info(f"Starting analysis of {file_path}")
        
        try:
            # 1. Document Processing
            logger.info("Extracting text from document...")
            raw_text = self.document_processor.extract_text(file_path)
            
            if not raw_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            # 2. Clause Extraction and Type Detection
            logger.info("Extracting and classifying clauses...")
            clauses = self.clause_extractor.extract_clauses(raw_text, document_id)
            
            # 3. Store clauses in vector database
            logger.info("Storing clauses in vector database...")
            for clause in tqdm(clauses, desc="Storing clauses"):
                self.vector_store.add_clause(clause)
            
            # 4. Risk Analysis
            logger.info("Analyzing risks...")
            for clause in tqdm(clauses, desc="Analyzing risks"):
                clause.risk_assessment = self.risk_analyzer.analyze_clause(clause)
            
            # 5. Generate Executive Summary
            logger.info("Generating executive summary...")
            executive_summary = self._generate_executive_summary(clauses)
            
            # 6. Processing Statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_stats = {
                'processing_time_seconds': processing_time,
                'total_clauses': len(clauses),
                'characters_processed': len(raw_text),
                'avg_clause_length': np.mean([len(c.text) for c in clauses]),
                'clause_types_detected': len(set(c.metadata.clause_type for c in clauses))
            }
            
            # 7. Create Analysis Result
            analysis = ContractAnalysis(
                document_id=document_id,
                clauses=clauses,
                executive_summary=executive_summary,
                generated_at=datetime.now(),
                processing_stats=processing_stats
            )
            
            logger.info(f"Analysis completed in {processing_time:.2f} seconds")
            return analysis
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            raise
    
    def _generate_executive_summary(self, clauses: List[ContractClause]) -> Dict[str, Any]:
        """Generate executive summary of the analysis"""
        # Risk distribution
        risk_distribution = Counter()
        total_priority_score = 0
        
        for clause in clauses:
            if clause.risk_assessment:
                risk_distribution[clause.risk_assessment.risk_level] += 1
                total_priority_score += clause.risk_assessment.priority_score
        
        # Key findings
        key_findings = []
        
        if risk_distribution['HIGH'] > 0:
            key_findings.append(f"{risk_distribution['HIGH']} high-risk clauses identified requiring immediate attention")
        
        if risk_distribution['MEDIUM'] > 0:
            key_findings.append(f"{risk_distribution['MEDIUM']} medium-risk clauses should be reviewed")
        
        # Clause type analysis
        clause_types = Counter(c.metadata.clause_type for c in clauses)
        most_common_type = clause_types.most_common(1)[0] if clause_types else ("general", 0)
        key_findings.append(f"Most common clause type: {most_common_type[0]} ({most_common_type[1]} instances)")
        
        # High priority clauses
        high_priority_clauses = [c for c in clauses 
                               if c.risk_assessment and c.risk_assessment.priority_score > 2.0]
        if high_priority_clauses:
            key_findings.append(f"{len(high_priority_clauses)} clauses require priority review")
        
        return {
            'risk_distribution': dict(risk_distribution),
            'total_priority_score': total_priority_score,
            'key_findings': key_findings,
            'clause_type_distribution': dict(clause_types)
        }
    
    def generate_report(self, analysis: ContractAnalysis, output_format: str = 'html') -> str:
        """Generate professional report"""
        if output_format.lower() == 'html':
            return self.report_generator.generate_html_report(analysis)
        elif output_format.lower() == 'pdf':
            html_file = self.report_generator.generate_html_report(analysis)
            return self.report_generator.generate_pdf_report(html_file)
        else:
            raise ValueError("Supported formats: 'html', 'pdf'")
    
    def collect_user_feedback(self, clause: ContractClause, feedback: Dict) -> Dict:
        """Collect user feedback for continuous improvement"""
        return self.feedback_manager.collect_feedback(clause, feedback)
    
    def get_prioritized_review_list(self, analysis: ContractAnalysis) -> List[ContractClause]:
        """Get clauses prioritized for user review"""
        return self.feedback_manager.get_prioritized_clauses(analysis.clauses)
    
    def update_preferences(self) -> Dict:
        """Update system preferences based on feedback"""
        preferences = self.feedback_manager.infer_preferences()
        
        # Update risk registry with learned patterns
        if preferences:
            self._apply_preference_updates(preferences)
        
        return preferences
    
    def _apply_preference_updates(self, preferences: Dict):
        """Apply inferred preferences to improve future analyses"""
        # This could update the risk registry, prompt templates, etc.
        # For now, we'll log the preferences
        logger.info(f"Updated preferences: {preferences}")
        
        # Save preferences for future use
        data_dir = os.path.join(BASE_DATA_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        preferences_file = os.path.join(data_dir, 'user_preferences.json')
        with open(preferences_file, 'w') as f:
            json.dump(preferences, f, indent=2)
    
    def store_user_feedback(self, feedback_data: Dict) -> bool:
        """Store user feedback from web interface (wrapper for collect_user_feedback)"""
        try:
            # For now, let's create a simplified feedback storage that doesn't rely on complex vector search
            # This ensures feedback is always stored even if vector search has issues
            
            feedback_entry = {
                'document_id': feedback_data.get('document_id'),
                'clause_id': feedback_data.get('clause_id'),
                'feedback_type': feedback_data.get('feedback_type'),
                'feedback_text': feedback_data.get('feedback_text', ''),
                'risk_override': feedback_data.get('risk_override'),
                'timestamp': feedback_data.get('timestamp', datetime.now().isoformat()),
                'user_id': feedback_data.get('user_id', 'web_user')
            }
            
            # Store feedback in learning history
            self.feedback_manager.feedback_history.append(feedback_entry)
            self.feedback_manager._save_feedback_history()
            
            # Also try to add to vector store if possible
            try:
                fingerprint = f"{feedback_data.get('document_id')}_{feedback_data.get('clause_id')}"
                self.vector_store.add_feedback(fingerprint, feedback_entry)
            except Exception as ve:
                logger.warning(f"Could not add feedback to vector store: {ve}")
                # Continue anyway - feedback is still stored in history
            
            logger.info(f"Stored user feedback for clause {feedback_data.get('clause_id')} in document {feedback_data.get('document_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store user feedback: {e}")
            return False

# Example usage and testing
def main():
    """Example usage of the Contract Agent"""
    # Initialize agent
    agent = ContractAgent()
    
    # Example contract analysis
    try:
        # Replace with actual contract file path
        contract_path = "sample_contract.pdf"
        
        if os.path.exists(contract_path):
            # Analyze contract
            analysis = agent.analyze_contract(contract_path)
            
            # Generate reports
            html_report = agent.generate_report(analysis, 'html')
            pdf_report = agent.generate_report(analysis, 'pdf')
            
            print(f"Analysis completed!")
            print(f"HTML Report: {html_report}")
            print(f"PDF Report: {pdf_report}")
            
            # Get prioritized review list
            priority_clauses = agent.get_prioritized_review_list(analysis)
            print(f"\nTop {min(5, len(priority_clauses))} clauses for review:")
            
            for i, clause in enumerate(priority_clauses[:5], 1):
                print(f"{i}. {clause.metadata.clause_type} - {clause.risk_assessment.risk_level} "
                      f"(Priority: {clause.risk_assessment.priority_score:.2f})")
                print(f"   {clause.text[:100]}...")
                print()
            
            # Example feedback collection
            if priority_clauses:
                example_feedback = {
                    'risk_level': 'MEDIUM',  # User's corrected risk level
                    'reason': 'This clause is standard in our industry',
                    'confidence': 0.9
                }
                
                feedback_result = agent.collect_user_feedback(priority_clauses[0], example_feedback)
                print(f"Feedback collected: {feedback_result['timestamp']}")
                
                # Update preferences
                preferences = agent.update_preferences()
                print(f"Updated preferences: {preferences}")
        
        else:
            print(f"Sample contract file not found: {contract_path}")
            print("Please provide a valid contract file path to test the agent.")
    
    except Exception as e:
        print(f"Error during analysis: {e}")
        logger.error(f"Main execution error: {e}")

if __name__ == "__main__":
    main()