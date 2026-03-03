# Policy Document Dataset for Risk-Aware Clause Ranking

## Overview

This dataset contains 20 policy documents collected for the **Risk-Aware Policy Clause Importance Ranking** NLP project. The documents span five different policy categories and contain clauses with varying levels of risk severity.

**Created:** March 2, 2026 
**Total Documents:** 20  
**Total Estimated Clauses:** 800+  
**Format:** Plain text (.txt)

## Dataset Structure

```
policy_dataset/
├── raw/                          # Original policy documents
│   ├── policy_001_university_academic_integrity.txt
│   ├── policy_002_graduate_research_ethics.txt
│   ├── ... (18 more files)
│   └── policy_020_title_ix_policy.txt
├── processed/                    # (To be created by NLP pipeline)
│   └── segmented_clauses.csv
└── dataset_metadata.csv          # Document tracking information
```

## Document Categories

### 1. University Policies (4 documents)
- **policy_001**: Academic Integrity Policy
- **policy_002**: Graduate Research Ethics
- **policy_003**: Course Attendance Policy
- **policy_004**: Grade Appeal Policy

### 2. Terms of Service (4 documents)
- **policy_005**: Social Media Platform ToS
- **policy_006**: E-Commerce Marketplace ToS
- **policy_013**: Student Code of Conduct
- **policy_014**: SaaS Subscription Terms

### 3. Corporate Compliance (4 documents)
- **policy_007**: Data Protection Compliance (GDPR/CCPA)
- **policy_008**: Workplace Health & Safety
- **policy_015**: Insider Trading Prevention
- **policy_018**: Employee Social Media Policy

### 4. Healthcare Protocols (4 documents)
- **policy_009**: HIPAA Compliance
- **policy_010**: Medication Administration
- **policy_016**: Infection Control
- (Healthcare policies have particularly high-risk clauses)

### 5. Government Regulations (4 documents)
- **policy_011**: U.S. Export Control
- **policy_012**: GDPR Compliance
- **policy_017**: FDA Clinical Trial Regulations
- **policy_019**: Environmental Compliance
- **policy_020**: Title IX Policy

## Risk Level Distribution

Documents contain clauses spanning all four risk levels:

- **Low**: Informational, guidance, general statements
- **Medium**: Warnings, minor penalties, reporting obligations
- **High**: Suspensions, significant fines, license restrictions
- **Critical**: Expulsion, termination, criminal prosecution, regulatory action

## Key Features

### Rich in Risk Indicators
- Modal verbs (must, shall, may, should, required)
- Penalty keywords (expulsion, termination, fine, suspension)
- Consequence phrases (criminal charges, license revocation)
- Conditional structures (if, unless, provided that)
- Temporal constraints (within 24 hours, immediately)

### Domain Diversity
- Educational: Academic policies, student conduct
- Business: Terms of service, e-commerce, SaaS
- Healthcare: Patient safety, medical protocols
- Legal/Regulatory: Compliance requirements
- Corporate: HR policies, trading rules

### Realistic Structure
- Numbered sections and sub-sections
- Clear hierarchical organization
- Professional policy language
- Mix of specific and general clauses

## Usage Guidelines

1. **Reading Documents:**
```python
import os

docs_path = 'policy_dataset/raw/'
for filename in os.listdir(docs_path):
    if filename.endswith('.txt'):
        with open(os.path.join(docs_path, filename), 'r', encoding='utf-8') as f:
            text = f.read()
            # Process document
```

2. **Accessing Metadata:**
```python
import pandas as pd

metadata = pd.read_csv('policy_dataset/metadata/dataset_metadata.csv')
print(metadata.head())
```