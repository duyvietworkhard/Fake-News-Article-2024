import requests
from bs4 import BeautifulSoup
from langdetect import detect
import spacy
from underthesea import pos_tag


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


def detect_language(text):
    """
    Phát hiện ngôn ngữ của tiêu đề.
    """
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None


# Load English NLP model
nlp_en = spacy.load("en_core_web_sm")


def extract_keywords(title, lang):
    """
    Lấy những từ khoá từ tiêu đề bài viết, hỗ trợ cả tiếng Anh và tiếng Việt.
    """
    if lang == 'en':
        # Sử dụng spaCy để trích xuất từ khoá cho tiếng Anh
        doc = nlp_en(title)
        keywords = [chunk.text for chunk in doc.noun_chunks if chunk.root.dep_ in {'nsubj', 'dobj'}]
    elif lang == 'vi':
        # Sử dụng underthesea để trích xuất từ khoá cho tiếng Việt
        keywords = [word for word, pos in pos_tag(title) if pos in ['N', 'Np']]
    else:
        print("Unsupported language")
        return []

    return keywords


def google_fact_check_api(query):
    """
    Sử dụng Google Fact Check API để kiểm tra các luận chứng dựa trên các từ khoá trích được.
    """
    api_key = 'AIzaSyB-GkpCVQjPFiKLYaz0lY6qPqzasVdGcjE'  # Thay bằng Google API key của bạn
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

    # Step 2: Detect language of the title
    lang = detect_language(title)
    # if lang:
    #     results.append(["Detected Language:", lang])
    # else:
    #     results.append(["Unable to detect language."])

    # Step 3: Extract keywords from the title
    keywords = extract_keywords(title, lang)
    if not keywords:
        results.append(["Unable to extract keywords from the title."])
        return results

    results.append(["Extracted Keywords:", keywords])

    # Step 4: Use Google Fact Check API with extracted keywords
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
                title_review = review.get('title', 'No title')
                url_review = review.get('url', 'No URL')
                review_result = [f"{publisher_name} rating:", rating, f"Reviewed by {publisher_name}:", title_review,
                                 f"({url_review})"]
                claim_result.extend(review_result)

            fact_check_array.append(claim_result)

        results.append(fact_check_array)
    else:
        results.append(["No claims found."])

    return results


if __name__ == "__main__":
    # Ví dụ bài báo tiếng Việt
    article_url_vi = "https://vnexpress.net/tam-dung-khai-thac-san-bay-dong-hoi-de-tranh-bao-4794436.html"
    print("=== Fact Check Vietnamese Article ===")
    result_vi = fact_check_article(article_url_vi)
    for res in result_vi:
        print(res)

    print("\n=== Fact Check English Article ===")
    # Ví dụ bài báo tiếng Anh
    article_url_en = "https://www.statesman.com/story/news/politics/politifact/2024/08/04/politifact-claim-that-trump-would-cut-social-security-lacks-basis/74647925007/"
    result_en = fact_check_article(article_url_en)
    for res in result_en:
        print(res)