import requests
import textwrap

# Constants
API_ENDPOINT = "https://api.chat.consensus.app/search_papers"

# Function to search papers from chat.consensus.app
def search_papers(query: str, year_min: int = None, year_max: int = None, study_types: list = None, human: bool = False, sample_size_min: int = None, sjr_max: int = None) -> list:
    payload = {
        "query": query,
        "year_min": year_min,
        "year_max": year_max,
        "study_types": study_types,
        "human": human,
        "sample_size_min": sample_size_min,
        "sjr_max": sjr_max
    }
    response = requests.post(API_ENDPOINT, json=payload)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        raise Exception(f"Error in API request: {response.status_code}")

# Function to synthesize information from papers
def synthesize_information(papers: list) -> tuple:
    if not papers:
        return "No relevant papers found.", [], "Unable to draw conclusions without relevant data."

    # This is a simplified placeholder for complex NLP tasks
    introduction = "Recent studies on the topic have revealed several key insights:"
    evidence = [f"{paper['paper_title']} ({paper['paper_publish_year']}) - {textwrap.shorten(paper['abstract'], 150)} [{paper['consensus_paper_details_url']}]"
                for paper in papers]
    conclusion = "These studies collectively enhance our understanding of the topic."

    return introduction, evidence, conclusion

# Function to format response
def format_response(introduction: str, evidence: list, conclusion: str) -> str:
    formatted_evidence = "\n".join(evidence)
    return f"{introduction}\n\n{formatted_evidence}\n\nConclusion: {conclusion}"

# Example usage
def main():
    query = "What are effective ways to reduce homelessness?"
    try:
        papers = search_papers(query)
        intro, evidence, concl = synthesize_information(papers)
        response = format_response(intro, evidence, concl)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
