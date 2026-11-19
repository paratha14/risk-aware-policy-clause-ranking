from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Requires the GOOGLE_API_KEY environment variable to be set
lm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

prompt = PromptTemplate.from_template(
    """You are a legal document parsing and clause extraction engine.
Your task is to extract ALL valid clauses from the supplied employment agreement and return them as a simple list.

Instructions
1. Identify every clause, section, subsection, and subclause.
2. Extract the exact original text of each clause. Do not summarize, paraphrase, rewrite, or correct it.
3. If a clause has a heading, keep it together with that clause's text.
4. Keep a complete clause as one item even when it spans multiple paragraphs or pages.
5. Do not split a single clause merely because it contains multiple sentences or paragraphs.
6. Do not merge separate clauses just because they discuss the same subject.
7. Do not invent any content.
8. Extract all clauses, including administrative, procedural, financial, and legal provisions. Do not omit clauses because they appear unimportant.

Strict inclusion rules
9. Include ONLY text that is part of a clause, section, subsection, or subclause of the agreement.
10. Do NOT include text that is not a clause, such as: cover pages, abstracts, summaries, tables of contents, stray notes, watermarks, running headers or footers, page numbers, letterheads, stamps, annexure index lists, signature blocks, witness details, and any random or unrelated text.
11. Do NOT include running headers, footers, or page numbers inside a clause's text, even if they appear in the middle of that clause across a page break.
12. When unsure whether a piece of text is a clause or non-clause text, leave it out.

Output
Return only a valid JSON list of strings, where each string is one clause's exact text, in the order they appear in the document:
["clause 1 text", "clause 2 text", "clause 3 text"]
Return only valid JSON. Do not perform risk classification or provide legal advice.

Employment agreement text:
{text}
"""
)


def segment_clauses(text: str) -> list:
    chain = prompt | lm | JsonOutputParser()
    return chain.invoke({"text": text})

if __name__ == "__main__":
    sample_text = """
    1. Introduction
    This Employment Agreement ("Agreement") is made and entered into as of the date of acceptance by the Employee.

    2. Position and Duties
    2.1 Position
    The Employee shall serve as a Software Engineer.
    2.2 Duties
    The Employee shall perform all duties as may be assigned by the Employer.

    3. Compensation
    3.1 Salary
    The Employee shall receive a monthly salary of ₹40,000.
    3.2 Benefits
    The Employee shall be entitled to health insurance and other benefits as per company policy.

    4. Termination
    4.1 Termination by Employer
    The Employer may terminate this Agreement for cause with immediate effect.
    4.2 Termination by Employee
    The Employee may terminate this Agreement with a notice period of one month.

    5. Confidentiality
    The Employee agrees to maintain the confidentiality of all proprietary information.

    6. Governing Law
    This Agreement shall be governed by and construed in accordance with the laws of India.
    """

    result = segment_clauses(sample_text)
    print(result)