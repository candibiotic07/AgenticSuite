"""
Contract Agent Demo Script

This script demonstrates how to use the Contract Review & Risk-Flagging Agent
with sample data and provides setup instructions.
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contract_agent import ContractAgent, logger

def setup_environment():
    """Setup environment and check requirements"""
    print("🔧 Setting up Contract Agent environment...")
    
    # Check if Google API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("\nOr create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here")
        return False
    
    print("✅ Google API key found")
    
    # Create sample contracts directory (main data directory is handled by contract_agent.py)
    sample_dir = os.path.join("contractDATAtemp", "sample_contracts")
    os.makedirs(sample_dir, exist_ok=True)
    print(f"✅ Sample contracts directory ready: {sample_dir}")
    
    return True

def create_sample_contract():
    """Create a sample contract for testing"""
    sample_contract_path = os.path.join("contractDATAtemp", "sample_contracts", "sample_agreement.txt")
    
    if os.path.exists(sample_contract_path):
        return sample_contract_path
    
    sample_content = """
SAMPLE SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into on [DATE] between Company A ("Client") and Company B ("Service Provider").

1. PAYMENT TERMS
The Client agrees to pay the Service Provider $50,000 within 15 days of invoice receipt. Late payments will incur a penalty of 2% per month. Payment must be made in advance for all services.

2. LIABILITY AND INDEMNIFICATION  
The Service Provider shall have unlimited liability for any damages arising from this agreement. The Client agrees to indemnify and hold harmless the Service Provider from any claims. Consequential damages may apply without limitation.

3. TERMINATION CLAUSE
Either party may terminate this agreement immediately without cause and without notice. Upon termination, all payments made are non-refundable.

4. INTELLECTUAL PROPERTY
All intellectual property rights, including patents, copyrights, and trade secrets, shall be exclusively owned by the Service Provider. The Client waives all rights to any work product created under this agreement.

5. CONFIDENTIALITY
The Client agrees to maintain strict confidentiality of all proprietary information. Breach of confidentiality will result in liquidated damages of $100,000.

6. GOVERNING LAW
This agreement shall be governed by the laws of [JURISDICTION]. All disputes must be resolved through binding arbitration with no right to appeal.

7. WARRANTIES
The Service Provider provides no warranties of any kind. All services are provided "as is" without guarantee of performance or results.

8. FORCE MAJEURE
The Service Provider shall not be liable for delays due to force majeure events including but not limited to acts of God, natural disasters, or government actions.
"""
    
    with open(sample_contract_path, 'w') as f:
        f.write(sample_content)
    
    print(f"✅ Created sample contract: {sample_contract_path}")
    return sample_contract_path

def demonstrate_basic_analysis():
    """Demonstrate basic contract analysis"""
    print("\n📄 Starting Basic Contract Analysis Demo...")
    
    try:
        # Initialize agent
        agent = ContractAgent()
        print("✅ Contract Agent initialized")
        
        # Create sample contract
        contract_path = create_sample_contract()
        
        # Analyze contract
        print(f"🔍 Analyzing contract: {contract_path}")
        analysis = agent.analyze_contract(contract_path, "sample_agreement")
        
        # Print basic results
        print(f"\n📊 Analysis Results:")
        print(f"   • Total clauses: {len(analysis.clauses)}")
        print(f"   • Processing time: {analysis.processing_stats['processing_time_seconds']:.2f} seconds")
        
        # Show risk distribution
        risk_dist = analysis.executive_summary['risk_distribution']
        print(f"\n⚠️  Risk Distribution:")
        for level, count in risk_dist.items():
            print(f"   • {level}: {count} clauses")
        
        # Show top risky clauses
        risky_clauses = [c for c in analysis.clauses 
                        if c.risk_assessment and c.risk_assessment.risk_level in ['HIGH', 'MEDIUM']]
        risky_clauses.sort(key=lambda x: x.risk_assessment.priority_score, reverse=True)
        
        print(f"\n🚨 Top Risky Clauses:")
        for i, clause in enumerate(risky_clauses[:3], 1):
            print(f"   {i}. {clause.metadata.clause_type.upper()} - {clause.risk_assessment.risk_level}")
            print(f"      Priority Score: {clause.risk_assessment.priority_score:.2f}")
            print(f"      Rationale: {clause.risk_assessment.rationale[:100]}...")
            print()
        
        return analysis, agent
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Demo analysis failed: {e}")
        return None, None

def demonstrate_report_generation(analysis, agent):
    """Demonstrate report generation"""
    if not analysis:
        return
    
    print("\n📋 Generating Reports...")
    
    try:
        # Generate HTML report
        html_report = agent.generate_report(analysis, 'html')
        print(f"✅ HTML report generated: {html_report}")
        
        # Generate PDF report
        try:
            pdf_report = agent.generate_report(analysis, 'pdf')
            if "enhanced_fallback" in pdf_report:
                print(f"✅ Enhanced PDF report generated (using ReportLab fallback): {pdf_report}")
                print("   💡 Note: Full PDF features require WeasyPrint. See installation guide below.")
            elif "simple_fallback" in pdf_report:
                print(f"⚠️  Simple PDF report generated (basic fallback): {pdf_report}")
                print("   💡 Note: For better PDF reports, install WeasyPrint properly. See guide below.")
            else:
                print(f"✅ Full PDF report generated using WeasyPrint: {pdf_report}")
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            print("   HTML report is still available for review")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")

def demonstrate_feedback_loop(analysis, agent):
    """Demonstrate feedback collection and learning"""
    if not analysis:
        return
    
    print("\n🔄 Demonstrating Feedback Loop...")
    
    try:
        # Get prioritized clauses for review
        priority_clauses = agent.get_prioritized_review_list(analysis)
        
        if priority_clauses:
            print(f"📝 Top priority clause for review:")
            top_clause = priority_clauses[0]
            print(f"   Type: {top_clause.metadata.clause_type}")
            print(f"   Risk: {top_clause.risk_assessment.risk_level}")
            print(f"   Priority: {top_clause.risk_assessment.priority_score:.2f}")
            print(f"   Text: {top_clause.text[:150]}...")
            
            # Simulate user feedback
            print(f"\n💬 Simulating user feedback...")
            example_feedback = {
                'risk_level': 'MEDIUM',  # User correction
                'reason': 'This is standard in our industry contracts',
                'confidence': 0.85,
                'notes': 'Common clause, not as risky as initially assessed'
            }
            
            feedback_result = agent.collect_user_feedback(top_clause, example_feedback)
            print(f"✅ Feedback collected at: {feedback_result['timestamp']}")
            
            # Update preferences
            preferences = agent.update_preferences()
            print(f"✅ Preferences updated: {len(preferences)} categories learned")
            
    except Exception as e:
        print(f"❌ Feedback demo failed: {e}")

def demonstrate_advanced_features():
    """Demonstrate advanced features"""
    print("\n🚀 Advanced Features Overview:")
    
    features = [
        "✅ Hybrid Analysis: Rule-based + LLM reasoning",
        "✅ Self-Consistency: Multiple analysis iterations with majority voting", 
        "✅ Semantic Retrieval: ChromaDB vector search for similar clauses",
        "✅ Confidence-Based Prioritization: Focus on uncertain decisions",
        "✅ Clause Type Detection: Automatic categorization (8 types)",
        "✅ Professional Reports: HTML/PDF with executive summaries",
        "✅ Feedback Learning: PRELUDE-style preference inference",
        "✅ Risk Registry: JSON-based rule patterns (automatically created)",
        "✅ OCR Fallback: Handle scanned documents",
        "✅ Multi-format Support: PDF, DOCX, TXT"
    ]
    
    for feature in features:
        print(f"   {feature}")

def show_usage_instructions():
    """Show detailed usage instructions"""
    print("\n📖 Usage Instructions:")
    print("""
1. Basic Usage:
   from contract_agent import ContractAgent
   
   agent = ContractAgent()
   analysis = agent.analyze_contract('contract.pdf')
   html_report = agent.generate_report(analysis, 'html')

2. Collect Feedback:
   priority_clauses = agent.get_prioritized_review_list(analysis)
   feedback = {'risk_level': 'MEDIUM', 'reason': 'Standard clause'}
   agent.collect_user_feedback(priority_clauses[0], feedback)
   
3. Update Learning:
   preferences = agent.update_preferences()

4. Advanced Configuration:
   # All data is automatically organized in contractDATAtemp/
   agent = ContractAgent()  # Uses default clean directory structure
   
   # Generate specific format reports  
   pdf_report = agent.generate_report(analysis, 'pdf')

5. Environment Setup:
   export GEMINI_API_KEY='your_gemini_api_key'
   pip install -r requirements.txt

6. WeasyPrint Setup (for full PDF reports):
   For full-featured PDF reports, WeasyPrint requires additional setup:
   
   Windows:
   - Install GTK+ libraries: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
   - Or use: conda install -c conda-forge weasyprint
   
   macOS:
   - brew install cairo pango gdk-pixbuf libffi
   - pip install WeasyPrint
   
   Linux:
   - sudo apt-get install build-essential python3-dev python3-pip python3-setuptools python3-wheel python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
   - pip install WeasyPrint
   
   Note: If WeasyPrint setup fails, the system will use an enhanced ReportLab fallback.
""")

def main():
    """Main demo function"""
    print("🤖 Contract Review & Risk-Flagging Agent Demo")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        return
    
    # Run demonstrations
    analysis, agent = demonstrate_basic_analysis()
    demonstrate_report_generation(analysis, agent)
    demonstrate_feedback_loop(analysis, agent)
    demonstrate_advanced_features()
    show_usage_instructions()
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Set up your Gemini API key")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run analysis on your own contracts")
    print("4. Review generated reports and provide feedback")
    print("5. Watch the system improve over time!")

if __name__ == "__main__":
    main()