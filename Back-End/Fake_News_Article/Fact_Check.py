import requests
from bs4 import BeautifulSoup
import spacy


def get_article_title(url):
    """
    Trích lấy tiêu đề của bài viết từ đường dẫn.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')

        # Try different common title tags
        title = soup.title.string if soup.title else None
        if not title:
            title_tag = soup.find('h1')  # Some articles use <h1> as the title
            title = title_tag.string if title_tag else None

        return title.strip() if title else None

    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None


# Load English NLP model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(title):
    """
    Lấy những từ khoá từ tiêu đề bài viết.
    """
    doc = nlp(title)
    keywords = [chunk.text for chunk in doc.noun_chunks if chunk.root.dep_ in {'nsubj', 'dobj'}]
    return keywords


def google_fact_check_api(query):
    """
    Sử dụng Google để kiểm tra các luận chứng dựa trên các từ khoá trích được.
    """
    api_key = 'AIzaSyB-GkpCVQjPFiKLYaz0lY6qPqzasVdGcjE'  # Replace with your Google API key
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={api_key}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error from Google Fact Check API: {response.status_code}")
        return None


def fact_check_article(url):
    results = []

    # Step 1: Get the title of the article
    title = get_article_title(url)
    if not title:
        results.append(["Unable to retrieve the title from the provided URL."])
        return results

    results.append(["Article Title:", title])

    # Step 2: Extract keywords from the title
    keywords = extract_keywords(title)
    if not keywords:
        results.append(["Unable to extract keywords from the title."])
        return results

    results.append(["Extracted Keywords:", keywords])

    # Step 3: Use Google Fact Check API with extracted keywords
    fact_check_results = google_fact_check_api(" ".join(keywords))
    if not fact_check_results:
        results.append(["No fact-checking results found."])
        return results

    # Display results
    claims = fact_check_results.get('claims', [])
    if claims:
        fact_check_array = ["Fact-Check Results:"]
        for claim in claims:
            text = claim.get('text', 'No text available')
            claimant = claim.get('claimant', 'No claimant')
            rating = claim.get("claimReview", [{}])[0].get("textualRating", "No rating available")
            claim_date = claim.get('claimDate', 'No date')
            claim_result = [f"{text} by {claimant} on {claim_date}"]

            for review in claim.get('claimReview', []):
                publisher = review.get('publisher', {})
                publisher_name = publisher.get('name', 'Unknown publisher')
                title = review.get('title', 'No title')
                url = review.get('url', 'No URL')
                review_result = [f"{publisher_name} rating:", rating, f"Reviewed by {publisher_name}:", title, f"({url})"]
                claim_result.extend(review_result)

            fact_check_array.append(claim_result)

        results.append(fact_check_array)
    else:
        results.append(["No claims found."])

    return results




if __name__ == "__main__":
    article_url = "https://www.statesman.com/story/news/politics/politifact/2024/08/04/politifact-claim-that-trump-would-cut-social-security-lacks-basis/74647925007/"
    fact_check_article(article_url)
